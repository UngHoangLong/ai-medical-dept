"""
Merged FastAPI backend:
- AI Medical Department API (/api/v1/analyze, patient history routes, etc.)
- VoxTell CT Viewer endpoints using patient-based S3 DICOM source:
    GET  /voxtell/volume/{pid}/{series_uid}
    POST /voxtell/predict

Run examples from repo root:
  uvicorn serving.backend.server:app --host 0.0.0.0 --port 1711
  python -m serving.backend.server
"""

import gc
import glob
import logging
import os
import sys
import shutil
import subprocess
import tempfile
import zipfile
from typing import Annotated

# Reduce CUDA memory fragmentation before torch is initialized
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# This file lives in serving/backend. Add the repo root to sys.path so
# both direct execution and uvicorn module execution work on Windows/Git Bash.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import boto3
import pydicom
import torch
import uvicorn
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from starlette.background import BackgroundTask

from serving.backend.pipeline import MedicalPipeline
from serving.backend.routes import router
from voxtell.inference.predictor import VoxTellPredictor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("serving.backend.pipeline").setLevel(logging.DEBUG)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("merged.backend")

app = FastAPI(title="AI Medical Department + VoxTell", version="0.1.0")

# If CORS_ORIGINS is not set, allow all origins but do not use credentials with wildcard.
# Example explicit value:
#   CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,https://ai-medical-dept.vercel.app
raw_origins = os.getenv("CORS_ORIGINS", "*")
allow_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
allow_credentials = "*" not in allow_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_VOXTELL_MODEL_DIR = os.path.abspath(
    os.path.join(PROJECT_ROOT, "models", "voxtell_v1.1")
)
VOXTELL_MODEL_DIR = os.getenv("VOXTELL_MODEL_DIR", DEFAULT_VOXTELL_MODEL_DIR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "").strip()
AWS_REGION = os.getenv("AWS_REGION", "").strip() or None

predictor: VoxTellPredictor | None = None
_s3_client = None


@app.on_event("startup")
async def startup() -> None:
    """Load both the medical pipeline and VoxTell predictor once at startup."""
    global predictor

    app.state.pipeline = MedicalPipeline()
    logger.info("MedicalPipeline loaded.")

    if os.getenv("DISABLE_VOXTELL", "0") == "1":
        logger.warning("VoxTell loading disabled by DISABLE_VOXTELL=1.")
        predictor = None
        return

    logger.info("Loading VoxTell model from %s on %s...", VOXTELL_MODEL_DIR, DEVICE)
    try:
        predictor = VoxTellPredictor(model_dir=VOXTELL_MODEL_DIR, device=DEVICE)
        # Save VRAM during prediction
        predictor.perform_everything_on_device = False
        logger.info("VoxTell model loaded successfully.")
    except Exception as exc:
        predictor = None
        logger.exception("Error loading VoxTell model: %s", exc)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "voxtell": "loaded" if predictor is not None else "not_loaded",
    }


# Main AI Medical API routes, including POST /api/v1/analyze
app.include_router(router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# S3 + DICOM helpers
# ---------------------------------------------------------------------------

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
    """Build possible keys for the patient DICOM zip.

    Primary expected structure:
        dicom-raw/{pid}/{series_uid}.zip

    A fallback without .zip is kept because some DB rows display the key without
    the suffix in tooling, while the object may still be named exactly that way.
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
            logger.info("Downloading DICOM zip from s3://%s/%s", bucket, key)
            client.download_file(bucket, key, output_path)
            return key
        except ClientError as exc:
            last_error = exc
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                continue
            logger.exception("S3 download failed for key=%s", key)
            raise HTTPException(status_code=500, detail=f"Cannot download DICOM zip from S3: {exc}") from exc

    raise HTTPException(
        status_code=404,
        detail=(
            "DICOM zip not found in S3. Tried: "
            + ", ".join(_dicom_s3_key_candidates(pid, series_uid))
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
    result = subprocess.run(
        ["dcm2niix", "-z", "y", "-f", "converted", "-o", output_dir, dicom_dir],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"dcm2niix failed: {result.stderr}")

    nifti_files = glob.glob(os.path.join(output_dir, "*.nii.gz"))
    if not nifti_files:
        raise RuntimeError("dcm2niix produced no NIfTI output")

    return max(nifti_files, key=os.path.getsize)


def _patient_dicom_zip_to_nifti(pid: str, series_uid: str, work_dir: str) -> tuple[str, str]:
    """Download dicom-raw/{pid}/{series_uid}.zip from S3 and convert to NIfTI.

    Returns:
        (nifti_path, dicom_s3_key_used)
    """
    zip_path = os.path.join(work_dir, "dicom.zip")
    s3_key = _download_patient_dicom_zip(pid, series_uid, zip_path)

    extract_dir = os.path.join(work_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        _safe_extract_zip(zip_path, extract_dir)
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


# ---------------------------------------------------------------------------
# VoxTell patient-based endpoints
# ---------------------------------------------------------------------------

@app.get("/voxtell/volume/{pid}/{series_uid:path}")
async def get_voxtell_volume(pid: str, series_uid: str):
    """Return a temporary .nii.gz volume converted from S3 DICOM zip.

    Source object:
        s3://{S3_BUCKET_NAME}/dicom-raw/{pid}/{series_uid}.zip
    """
    pid = pid.strip()
    series_uid = series_uid.strip()
    if not pid or not series_uid:
        raise HTTPException(status_code=400, detail="pid and series_uid are required.")

    with tempfile.TemporaryDirectory() as temp_dir:
        nifti_path, s3_key = _patient_dicom_zip_to_nifti(pid, series_uid, temp_dir)

        final_path = os.path.join(
            tempfile.gettempdir(),
            f"voxtell_volume_{pid}_{os.urandom(8).hex()}.nii.gz",
        )
        shutil.copy(nifti_path, final_path)

    logger.info("Prepared VoxTell volume for pid=%s series=%s from key=%s", pid, series_uid, s3_key)

    return FileResponse(
        final_path,
        media_type="application/gzip",
        filename=f"{pid}_{series_uid}.nii.gz",
        background=BackgroundTask(_cleanup_file, final_path),
    )


@app.post("/voxtell/predict")
async def voxtell_predict(
    pid: Annotated[str, Form()],
    series_uid: Annotated[str, Form()],
    prompt: Annotated[str, Form()],
):
    """Run VoxTell segmentation for an existing patient series in S3.

    The frontend sends only pid, series_uid and prompt. Backend downloads the
    DICOM zip from S3, converts it to NIfTI internally, runs VoxTell, and returns
    the segmentation mask as .nii.gz. The mask is not persisted to S3.
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="VoxTell model not loaded.")

    pid = pid.strip()
    series_uid = series_uid.strip()
    prompt = prompt.strip()

    if not pid or not series_uid:
        raise HTTPException(status_code=400, detail="pid and series_uid are required.")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required.")

    with tempfile.TemporaryDirectory() as temp_dir:
        nifti_path, s3_key = _patient_dicom_zip_to_nifti(pid, series_uid, temp_dir)

        logger.info(
            "Running VoxTell segmentation pid=%s series=%s key=%s prompt=%r",
            pid,
            series_uid,
            s3_key,
            prompt,
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

            return FileResponse(
                final_output_path,
                media_type="application/gzip",
                filename=f"voxtell_{pid}_{safe_prompt}.nii.gz",
                background=BackgroundTask(_cleanup_file, final_output_path),
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Error during VoxTell inference: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    port = int(os.getenv("PORT", "1711"))
    uvicorn.run(app, host="0.0.0.0", port=port)
