"""
Batch pipeline: IDC DICOM → NPY (float16) → HuggingFace Hub → xóa local.

- Resume tự động từ checkpoint nếu bị ngắt.
- Parallel workers (--workers) để download + upload song song.
- Checkpoint lưu tại scripts/ (bền vững, không mất khi restart).

Usage:
    python scripts/build_npy_hub.py \
        --keys-file  data/subset_keys/test_npy_keys.txt \
        --pid-map    data/subset_keys/test_npy_pid_map.json \
        --tmp-dir    /tmp/npy_test_build \
        --hf-repo    UngLong/openm3chest-npy-v2 \
        --workers    3 \
        [--checkpoint scripts/npy_test_checkpoint.txt]
"""

import argparse
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom
from huggingface_hub import HfApi
from idc_index import IDCClient
from tqdm import tqdm

_tls        = threading.local()
_init_lock  = threading.Lock()
_upload_sem = threading.Semaphore(2)  # 1 upload tại 1 lúc — tránh HF rate limit


# ---------------------------------------------------------------------------
# DICOM → NPY float16
# ---------------------------------------------------------------------------

def _sort_key(ds):
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])
    return int(getattr(ds, "InstanceNumber", 0))


def dicom_dir_to_npy(dicom_dir: Path, npy_out: Path) -> None:
    files = []
    for p in dicom_dir.rglob("*"):
        if p.is_file():
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=False)
                if hasattr(ds, "PixelData"):
                    files.append(ds)
            except Exception:
                pass
    if not files:
        raise RuntimeError(f"No DICOM slices in {dicom_dir}")
    files = sorted(files, key=_sort_key)
    slices = []
    for ds in files:
        arr       = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append((arr * slope + intercept).astype(np.float16))
    npy_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_out), np.stack(slices, axis=0))


# ---------------------------------------------------------------------------
# IDC client (thread-local)
# ---------------------------------------------------------------------------

def _get_idc_client():
    if not hasattr(_tls, "client"):
        with _init_lock:
            if not hasattr(_tls, "client"):
                _tls.client = IDCClient()
    return _tls.client


# ---------------------------------------------------------------------------
# Checkpoint (thread-safe)
# ---------------------------------------------------------------------------

def load_checkpoint(f: Path) -> set[str]:
    return set(f.read_text().splitlines()) if f.exists() else set()


def save_checkpoint(f: Path, done: set[str], lock: threading.Lock) -> None:
    with lock:
        f.write_text("\n".join(sorted(done)))


# ---------------------------------------------------------------------------
# Process one scan
# ---------------------------------------------------------------------------

def process_one(series_uid, pid, tmp_dir, hf_repo, api, idc_index,
                done_set, done_lock, checkpoint_file):
    with done_lock:
        if series_uid in done_set:
            return "already_done"

    match = idc_index[idc_index["SeriesInstanceUID"] == series_uid]
    if match.empty or match["Modality"].values[0] != "CT":
        with done_lock:
            done_set.add(series_uid)
        save_checkpoint(checkpoint_file, done_set, done_lock)
        return "skipped"

    dcm_dir  = tmp_dir / "_dicom" / pid / series_uid
    npy_path = tmp_dir / "npy"    / pid / f"{series_uid}.npy"
    dcm_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = _get_idc_client()
        client.download_dicom_series(
            seriesInstanceUID=series_uid,
            downloadDir=str(dcm_dir),
        )
        dicom_dir_to_npy(dcm_dir, npy_path)

        with _upload_sem:
            api.upload_file(
                path_or_fileobj=str(npy_path),
                path_in_repo=f"{pid}/{series_uid}.npy",
                repo_id=hf_repo,
                repo_type="dataset",
            )

        shutil.rmtree(dcm_dir, ignore_errors=True)
        npy_path.unlink(missing_ok=True)

        with done_lock:
            done_set.add(series_uid)
        save_checkpoint(checkpoint_file, done_set, done_lock)
        return "done"

    except Exception as e:
        shutil.rmtree(dcm_dir, ignore_errors=True)
        if npy_path.exists():
            npy_path.unlink()
        return f"failed:{e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    scripts_dir = Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--keys-file",  required=True,
                        help="File chứa series UIDs (1 dòng = 1 UID)")
    parser.add_argument("--pid-map",    required=True,
                        help="JSON: {series_uid: pid}")
    parser.add_argument("--tmp-dir",    required=True,
                        help="Thư mục tạm DICOM + NPY")
    parser.add_argument("--hf-repo",    required=True,
                        help="HuggingFace dataset repo")
    parser.add_argument("--checkpoint",
                        default=str(scripts_dir / "npy_upload_checkpoint.txt"),
                        help="File checkpoint (mặc định: scripts/npy_upload_checkpoint.txt)")
    parser.add_argument("--workers",    type=int, default=3,
                        help="Số parallel workers")
    parser.add_argument("--limit",      type=int,
                        help="Giới hạn số series xử lý (debug)")
    args = parser.parse_args()

    tmp_dir         = Path(args.tmp_dir)
    checkpoint_file = Path(args.checkpoint)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Load keys + pid map
    all_keys = Path(args.keys_file).read_text().splitlines()
    pid_map  = json.load(open(args.pid_map))
    if args.limit:
        all_keys = all_keys[:args.limit]

    # Resume
    done_set  = load_checkpoint(checkpoint_file)
    done_lock = threading.Lock()
    remaining = [k for k in all_keys if k not in done_set]

    print(f"Keys    : {len(all_keys)}")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"Đã xong : {len(done_set)} | Còn lại: {len(remaining)}")
    print(f"Workers : {args.workers}\n")

    if not remaining:
        print("Tất cả đã xong!")
        return

    # Init IDC + HF
    print("Initializing IDC client...")
    idc_index = IDCClient().index
    api = HfApi()
    try:
        api.create_repo(repo_id=args.hf_repo, repo_type="dataset",
                        exist_ok=True, private=True)
    except Exception:
        pass
    print(f"Repo: {args.hf_repo}\n")

    done_count = skipped = failed = 0
    pbar = tqdm(total=len(remaining), desc="Uploading NPY")

    executor = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = {
            executor.submit(
                process_one,
                uid,
                pid_map.get(uid, "unknown"),
                tmp_dir, args.hf_repo, api, idc_index,
                done_set, done_lock, checkpoint_file,
            ): uid
            for uid in remaining
        }
        for future in as_completed(futures):
            uid = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = f"failed:{e}"

            if result == "done":
                done_count += 1
            elif result == "skipped":
                skipped += 1
            elif result.startswith("failed"):
                failed += 1
                tqdm.write(f"[FAIL] {uid[:50]}: {result[7:80]}")
            pbar.update(1)
            pbar.set_postfix(done=done_count, skip=skipped, fail=failed)

    except KeyboardInterrupt:
        print("\n[STOP] Dừng lại — checkpoint đã lưu, chạy lại để resume.")
        executor.shutdown(cancel_futures=True, wait=False)
    finally:
        executor.shutdown(wait=False)

    pbar.close()
    print(f"\n{'='*50}")
    print(f"Done   : {done_count}")
    print(f"Skipped: {skipped} (non-CT)")
    print(f"Failed : {failed}")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"Hub    : https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()
