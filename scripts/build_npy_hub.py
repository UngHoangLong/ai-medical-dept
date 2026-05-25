"""
Batch pipeline: IDC DICOM → NPY (float16) → HuggingFace Hub → xóa local.

Tự động resume nếu bị ngắt giữa chừng.
NPY lưu dạng float16 (giảm 50% dung lượng, không ảnh hưởng chất lượng).
Cấu trúc trên Hub: {pid}/{series_uid}.npy — tương thích với DataPrepAgent.

Usage:
    python scripts/build_npy_hub.py \
        --data-json-dir data/data_json \
        --tmp-dir     /tmp/npy_build \
        --hf-repo     UngLong/openm3chest-npy \
        [--limit      2000]

Để dừng: Ctrl+C — lần sau chạy lại tự resume từ chỗ dừng.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pydicom
from huggingface_hub import HfApi
from idc_index import IDCClient
from tqdm import tqdm


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
        arr = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append((arr * slope + intercept).astype(np.float16))

    npy_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_out), np.stack(slices, axis=0))


# ---------------------------------------------------------------------------
# Build UID → PID mapping từ tất cả JSON files
# ---------------------------------------------------------------------------

def build_uid_pid_map(data_json_dir: Path) -> dict[str, str]:
    uid_pid = {}
    for json_file in sorted(data_json_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                records = json.load(f)
            for r in records:
                uid = str(r.get("keys", ""))
                pid = str(r.get("pids", "unknown"))
                if uid:
                    uid_pid[uid] = pid
        except Exception as e:
            print(f"[WARN] Skip {json_file.name}: {e}")
    return uid_pid


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_file: Path) -> set[str]:
    if not checkpoint_file.exists():
        return set()
    return set(checkpoint_file.read_text().splitlines())


def save_checkpoint(checkpoint_file: Path, done: set[str]) -> None:
    checkpoint_file.write_text("\n".join(sorted(done)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-json-dir", required=True, help="Thư mục chứa JSON label files")
    parser.add_argument("--tmp-dir",       required=True, help="Thư mục tạm để chứa DICOM + NPY")
    parser.add_argument("--hf-repo",       required=True, help="HuggingFace dataset repo, vd: UngLong/openm3chest-npy")
    parser.add_argument("--limit",         type=int,      help="Giới hạn số series cần xử lý (mặc định: tất cả)")
    args = parser.parse_args()

    data_json_dir  = Path(args.data_json_dir)
    tmp_dir        = Path(args.tmp_dir)
    checkpoint_file = tmp_dir / "checkpoint.txt"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- Build UID → PID map ---
    print("Building UID → PID map từ JSON files...")
    uid_pid = build_uid_pid_map(data_json_dir)
    all_uids = list(uid_pid.keys())
    if args.limit:
        all_uids = all_uids[:args.limit]
    print(f"Tổng: {len(uid_pid)} UIDs, sẽ xử lý: {len(all_uids)}")

    # --- Resume ---
    done = load_checkpoint(checkpoint_file)
    remaining = [u for u in all_uids if u not in done]
    print(f"Đã xong: {len(done)}, còn lại: {len(remaining)}\n")

    # --- Init IDC client ---
    print("Initializing IDC client...")
    client = IDCClient()
    df     = client.index

    # --- Init HuggingFace API ---
    api = HfApi()
    try:
        api.create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True, private=True)
        print(f"HuggingFace repo: {args.hf_repo}\n")
    except Exception as e:
        print(f"[WARN] Không tạo được repo (có thể đã tồn tại): {e}")

    # --- Process ---
    skipped = 0
    failed  = 0

    for series_uid in tqdm(remaining, desc="Processing"):
        pid = uid_pid[series_uid]

        # Check IDC
        match = df[df["SeriesInstanceUID"] == series_uid]
        if match.empty or match["Modality"].values[0] != "CT":
            skipped += 1
            done.add(series_uid)
            continue

        dcm_dir  = tmp_dir / "_dicom" / pid / series_uid
        npy_path = tmp_dir / "npy"    / pid / f"{series_uid}.npy"
        dcm_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Download DICOM
            client.download_dicom_series(
                seriesInstanceUID=series_uid,
                downloadDir=str(dcm_dir),
            )

            # 2. Convert → NPY float16
            dicom_dir_to_npy(dcm_dir, npy_path)

            # 3. Upload lên HuggingFace Hub
            api.upload_file(
                path_or_fileobj=str(npy_path),
                path_in_repo=f"{pid}/{series_uid}.npy",
                repo_id=args.hf_repo,
                repo_type="dataset",
            )

            # 4. Xóa local
            shutil.rmtree(dcm_dir, ignore_errors=True)
            npy_path.unlink(missing_ok=True)

            done.add(series_uid)
            save_checkpoint(checkpoint_file, done)

        except KeyboardInterrupt:
            print("\n[STOP] Dừng lại — checkpoint đã lưu, chạy lại để resume.")
            shutil.rmtree(dcm_dir, ignore_errors=True)
            break

        except Exception as e:
            print(f"\n[ERROR] {series_uid}: {e}")
            shutil.rmtree(dcm_dir, ignore_errors=True)
            if npy_path.exists():
                npy_path.unlink()
            failed += 1
            time.sleep(2)

    print(f"\n{'='*50}")
    print(f"Xong   : {len(done)}")
    print(f"Skipped: {skipped} (không phải CT trên IDC)")
    print(f"Failed : {failed}")
    print(f"Hub    : https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()
