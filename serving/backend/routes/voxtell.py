import gc
import glob
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from typing import Annotated

import boto3
import pydicom
import torch
from botocore.exceptions import ClientError
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from starlette.background import BackgroundTask

router = APIRouter(prefix="/voxtell", tags=["voxtell"])
logger = logging.getLogger(__name__)

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "").strip()
AWS_REGION = os.getenv("AWS_REGION", "").strip() or None
_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def _require_bucket() -> str:
    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="S3_BUCKET_NAME is not configured.")
    return S3_BUCKET_NAME


def _dicom_s3_key_candidates(pid: str, series_uid: str) -> list[str]:
    """Return S3 key candidates for patient DICOM zip.

    Primary expected structure:
        dicom-raw/{pid}/{series_uid}.zip
    """
    base = f"dicom-raw/{pid}/{series_uid}"
    if base.lower().endswith(".zip"):
        return [base]
    return [f"{base}.zip", base]


def _download_patient_dicom_zip(pid: str, series_uid: str, output_path: str) -> str:
    bucket = _require_bucket()
    client = _get_s3_client()
    last_error: Exception | None = None

    for key in _dicom_s3_key_candidates(pid, series_uid):
        try:
            logger.info("S3_GET_START bucket=%s key=%s", bucket, key)
            client.download_file(bucket, key, output_path)
            logger.info(
                "S3_GET_DONE bucket=%s key=%s path=%s size=%s",
                bucket,
                key,
                output_path,
                os.path.getsize(output_path),
            )
            return key
        except ClientError as exc:
            last_error = exc
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                continue
            logger.exception("S3_GET_FAILED bucket=%s key=%s", bucket, key)
            raise HTTPException(
                status_code=500,
                detail=f"Cannot download DICOM zip from S3: {exc}",
            ) from exc

    tried = ", ".join(_dicom_s3_key_candidates(pid, series_uid))
    raise HTTPException(
        status_code=404,
        detail=(
            f"DICOM zip not found in S3. Tried: {tried}"
            + (f". Last error: {last_error}" if last_error else "")
        ),
    )


def _safe_extract_zip(zip_path: str, extract_dir: str) -> None:
    """Extract zip safely, preventing path traversal."""
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        root = os.path.abspath(extract_dir)
        for member in zip_file.infolist():
            target = os.path.abspath(os.path.join(extract_dir, member.filename))
            if not target.startswith(root + os.sep) and target != root:
                raise HTTPException(status_code=400, detail="Invalid zip path detected.")
        zip_file.extractall(extract_dir)


def _find_dicom_dir(root: str) -> str | None:
    """Walk extracted zip to find a directory containing DICOM files."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname != "__MACOSX" and not dirname.startswith("._")
        ]

        for filename in filenames:
            if filename.startswith("._"):
                continue

            file_path = os.path.join(dirpath, filename)

            if filename.lower().endswith(".dcm"):
                return dirpath

            try:
                pydicom.dcmread(file_path, stop_before_pixels=True)
                return dirpath
            except Exception:
                continue

    return None


def convert_dicom_to_nifti(dicom_dir: str, output_dir: str) -> str:
    """Run dcm2niix and return the largest generated .nii.gz file."""
    logger.info("DCM2NIIX_START input=%s output=%s", dicom_dir, output_dir)
    result = subprocess.run(
        ["dcm2niix", "-z", "y", "-f", "converted", "-o", output_dir, dicom_dir],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error("DCM2NIIX_FAILED stderr=%s stdout=%s", result.stderr, result.stdout)
        raise RuntimeError(f"dcm2niix failed: {result.stderr}")

    nifti_files = glob.glob(os.path.join(output_dir, "*.nii.gz"))
    if not nifti_files:
        logger.error("DCM2NIIX_NO_OUTPUT stdout=%s stderr=%s", result.stdout, result.stderr)
        raise RuntimeError("dcm2niix produced no NIfTI output")

    nifti_path = max(nifti_files, key=os.path.getsize)
    logger.info("DCM2NIIX_DONE output=%s size=%s", nifti_path, os.path.getsize(nifti_path))
    return nifti_path


def _patient_dicom_zip_to_nifti(pid: str, series_uid: str, work_dir: str) -> tuple[str, str]:
    """Download dicom-raw/{pid}/{series_uid}.zip from S3 and convert to NIfTI."""
    zip_path = os.path.join(work_dir, "dicom.zip")
    s3_key = _download_patient_dicom_zip(pid, series_uid, zip_path)

    extract_dir = os.path.join(work_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        logger.info("UNZIP_START path=%s output=%s", zip_path, extract_dir)
        _safe_extract_zip(zip_path, extract_dir)
        logger.info("UNZIP_DONE output=%s", extract_dir)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Invalid DICOM zip file in S3.") from exc

    dicom_dir = _find_dicom_dir(extract_dir)
    if dicom_dir is None:
        raise HTTPException(status_code=400, detail="No DICOM files found in S3 zip.")

    nifti_out_dir = os.path.join(work_dir, "nifti")
    os.makedirs(nifti_out_dir, exist_ok=True)

    try:
        nifti_path = convert_dicom_to_nifti(dicom_dir, nifti_out_dir)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return nifti_path, s3_key


def _cleanup_file(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        logger.warning("Failed to remove temp file: %s", path, exc_info=True)


def _validate_patient_params(pid: str, series_uid: str) -> tuple[str, str]:
    pid = pid.strip()
    series_uid = series_uid.strip()
    if not pid or not series_uid:
        raise HTTPException(status_code=400, detail="pid and series_uid are required.")
    return pid, series_uid


@router.get("/volume/{pid}/{series_uid:path}")
async def get_voxtell_volume(pid: str, series_uid: str):
    """Return .nii.gz volume converted from S3 DICOM zip.

    Source object:
        s3://{S3_BUCKET_NAME}/dicom-raw/{pid}/{series_uid}.zip
    """
    pid, series_uid = _validate_patient_params(pid, series_uid)

    with tempfile.TemporaryDirectory() as temp_dir:
        nifti_path, s3_key = _patient_dicom_zip_to_nifti(pid, series_uid, temp_dir)

        final_path = os.path.join(
            tempfile.gettempdir(),
            f"voxtell_volume_{pid}_{os.urandom(8).hex()}.nii.gz",
        )
        shutil.copy(nifti_path, final_path)

    logger.info("VOLUME_READY pid=%s series=%s key=%s file=%s", pid, series_uid, s3_key, final_path)

    return FileResponse(
        final_path,
        media_type="application/gzip",
        filename=f"{pid}_{series_uid}.nii.gz",
        background=BackgroundTask(_cleanup_file, final_path),
    )


@router.post("/predict")
async def voxtell_predict(
    request: Request,
    pid: Annotated[str, Form()],
    series_uid: Annotated[str, Form()],
    prompt: Annotated[str, Form()],
):
    """Run VoxTell segmentation for a patient series stored in S3.

    Request fields:
        pid, series_uid, prompt

    Internal flow:
        S3 DICOM zip -> dcm2niix NIfTI -> VoxTell -> segmentation .nii.gz
    """
    predictor = getattr(request.app.state, "voxtell_predictor", None)
    if predictor is None:
        raise HTTPException(status_code=500, detail="VoxTell model not loaded.")

    pid, series_uid = _validate_patient_params(pid, series_uid)
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required.")

    with tempfile.TemporaryDirectory() as temp_dir:
        nifti_path, s3_key = _patient_dicom_zip_to_nifti(pid, series_uid, temp_dir)

        logger.info(
            "VOXTELL_PREDICT_START pid=%s series=%s key=%s prompt=%r nifti=%s",
            pid,
            series_uid,
            s3_key,
            prompt,
            nifti_path,
        )

        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            reader_writer = NibabelIOWithReorient()
            img, props = reader_writer.read_images([nifti_path])

            segmentations = predictor.predict_single_image(img, [prompt])
            seg_result = segmentations[0]

            output_path = os.path.join(temp_dir, "segmentation.nii.gz")
            reader_writer.write_seg(seg_result, output_path, props)

            safe_prompt = "_".join(prompt.split())[:80] or "mask"
            final_output_path = os.path.join(
                tempfile.gettempdir(),
                f"voxtell_segmentation_{pid}_{os.urandom(8).hex()}.nii.gz",
            )
            shutil.copy(output_path, final_output_path)

            logger.info(
                "VOXTELL_PREDICT_DONE pid=%s series=%s prompt=%r output=%s size=%s",
                pid,
                series_uid,
                prompt,
                final_output_path,
                os.path.getsize(final_output_path),
            )

            return FileResponse(
                final_output_path,
                media_type="application/gzip",
                filename=f"voxtell_{pid}_{safe_prompt}.nii.gz",
                background=BackgroundTask(_cleanup_file, final_output_path),
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("VOXTELL_PREDICT_FAILED: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
