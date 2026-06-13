"""
Merged FastAPI backend:
- AI Medical Department API (/api/v1/analyze, patient history routes, etc.)
- VoxTell CT Viewer endpoints (/predict, /convert, /export-rtstruct, /session/{session_id})

Run examples from repo root:
  uvicorn serving.backend.server:app --host 0.0.0.0 --port 1711
  python -m serving.backend.server
"""

import gc
import glob
import json
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

import nibabel as nib
import numpy as np
import pydicom
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from rt_utils import RTStructBuilder

from serving.backend.pipeline import MedicalPipeline
from serving.backend.routes import router
from voxtell.inference.predictor import VoxTellPredictor

# Works whether this file is placed at repo root or inside serving/backend
try:
    from dicom_sessions import (
        cleanup_expired,
        cleanup_session,
        create_session,
        get_session_dicom_dir,
    )
except ImportError:
    from serving.backend.dicom_sessions import (
        cleanup_expired,
        cleanup_session,
        create_session,
        get_session_dicom_dir,
    )

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
#   CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
raw_origins = os.getenv("CORS_ORIGINS", "*")
allow_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
allow_credentials = "*" not in allow_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)

DEFAULT_VOXTELL_MODEL_DIR = os.path.abspath(
    os.path.join(PROJECT_ROOT, "models", "voxtell_v1.1")
)
VOXTELL_MODEL_DIR = os.getenv("VOXTELL_MODEL_DIR", DEFAULT_VOXTELL_MODEL_DIR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predictor: VoxTellPredictor | None = None


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
# VoxTell DICOM helpers
# ---------------------------------------------------------------------------

def _reorient_nifti_mask_to_dicom(nii_path: str, sorted_series_data: list) -> np.ndarray:
    """Reorient a NIfTI mask to match the DICOM pixel grid expected by rt-utils.

    rt-utils expects mask shape (Columns, Rows, num_slices) where axes correspond
    to the DICOM pixel grid, not the NIfTI RAS voxel grid.
    dcm2niix reorients to RAS, so this computes axis permutation and flips from
    the NIfTI affine and DICOM geometry.
    """
    nii = nib.load(nii_path)
    mask = np.asanyarray(nii.dataobj).astype(bool)
    affine = nii.affine

    ref_dcm = sorted_series_data[0]

    iop = np.array(ref_dcm.ImageOrientationPatient, dtype=float)
    row_cosine_lps = iop[:3]   # direction of increasing column index
    col_cosine_lps = iop[3:]   # direction of increasing row index

    if len(sorted_series_data) > 1:
        pos0 = np.array(sorted_series_data[0].ImagePositionPatient, dtype=float)
        pos1 = np.array(sorted_series_data[1].ImagePositionPatient, dtype=float)
        slice_dir_lps = pos1 - pos0
        slice_dir_lps = slice_dir_lps / np.linalg.norm(slice_dir_lps)
    else:
        slice_dir_lps = np.cross(row_cosine_lps, col_cosine_lps)

    lps_to_ras = np.array([-1, -1, 1], dtype=float)

    dicom_axes_ras = np.column_stack(
        [
            col_cosine_lps * lps_to_ras,
            row_cosine_lps * lps_to_ras,
            slice_dir_lps * lps_to_ras,
        ]
    )

    nii_axes_ras = np.zeros((3, 3))
    for ax in range(3):
        vector = affine[:3, ax]
        nii_axes_ras[:, ax] = vector / np.linalg.norm(vector)

    corr = nii_axes_ras.T @ dicom_axes_ras

    perm: list[int] = []
    flips: list[bool] = []
    for dicom_ax in range(3):
        abs_corr = np.abs(corr[:, dicom_ax])
        nii_ax = int(np.argmax(abs_corr))
        perm.append(nii_ax)
        flips.append(bool(corr[nii_ax, dicom_ax] < 0))

    result = np.transpose(mask, perm)
    for ax, should_flip in enumerate(flips):
        if should_flip:
            result = np.flip(result, axis=ax)

    return np.ascontiguousarray(result)


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


# ---------------------------------------------------------------------------
# VoxTell endpoints
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict(
    image: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()],
):
    if predictor is None:
        raise HTTPException(status_code=500, detail="VoxTell model not loaded.")

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, image.filename or "input.nii.gz")

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        logger.info("Processing %s with prompt: %r", image.filename, prompt)

        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            reader_writer = NibabelIOWithReorient()
            img, props = reader_writer.read_images([input_path])

            segmentations = predictor.predict_single_image(img, [prompt])
            seg_result = segmentations[0]

            output_filename = f"segmentation_{image.filename or 'input.nii.gz'}"
            output_path = os.path.join(temp_dir, output_filename)
            reader_writer.write_seg(seg_result, output_path, props)

            final_output_path = os.path.join(
                tempfile.gettempdir(),
                f"voxtell_output_{os.urandom(8).hex()}.nii.gz",
            )
            shutil.copy(output_path, final_output_path)

            return FileResponse(
                final_output_path,
                media_type="application/gzip",
                filename=output_filename,
                background=None,
            )

        except Exception as exc:
            logger.exception("Error during VoxTell inference: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/convert")
async def convert_dicom(file: Annotated[UploadFile, File()]):
    """Accept a zipped DICOM folder, convert to NIfTI, return file + session ID."""
    cleanup_expired()

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a .zip file containing DICOM data.",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extract_dir = os.path.join(temp_dir, "extracted")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                zip_file.extractall(extract_dir)
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="Invalid zip file.") from exc

        dicom_dir = _find_dicom_dir(extract_dir)
        if dicom_dir is None:
            raise HTTPException(
                status_code=400,
                detail="No DICOM files found in the uploaded zip.",
            )

        session_id = create_session(dicom_dir)

        nifti_out_dir = os.path.join(temp_dir, "nifti")
        os.makedirs(nifti_out_dir, exist_ok=True)

        try:
            nifti_path = convert_dicom_to_nifti(dicom_dir, nifti_out_dir)
        except RuntimeError as exc:
            cleanup_session(session_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        final_path = os.path.join(
            tempfile.gettempdir(),
            f"voxtell_converted_{os.urandom(8).hex()}.nii.gz",
        )
        shutil.copy(nifti_path, final_path)

        logger.info("DICOM converted: session=%s, nifti=%s", session_id, final_path)

        response = FileResponse(
            final_path,
            media_type="application/gzip",
            filename="converted.nii.gz",
            background=None,
        )
        response.headers["X-Session-Id"] = session_id
        return response


@app.post("/export-rtstruct")
async def export_rtstruct(
    session_id: Annotated[str, Form()],
    structure_names: Annotated[str, Form()],
    segmentation_files: list[UploadFile] = File(...),
):
    """Build an RTSTRUCT from stored DICOM series and uploaded masks."""
    dicom_dir = get_session_dicom_dir(session_id)
    if dicom_dir is None:
        raise HTTPException(status_code=404, detail="DICOM session not found or expired.")

    try:
        names: list[str] = json.loads(structure_names)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="structure_names must be a JSON array of strings.",
        ) from exc

    if len(names) != len(segmentation_files):
        raise HTTPException(
            status_code=400,
            detail=f"Got {len(names)} names but {len(segmentation_files)} segmentation files.",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        rtstruct = RTStructBuilder.create_new(dicom_series_path=dicom_dir)

        for seg_file, name in zip(segmentation_files, names):
            seg_path = os.path.join(temp_dir, seg_file.filename or "seg.nii.gz")
            with open(seg_path, "wb") as buffer:
                shutil.copyfileobj(seg_file.file, buffer)

            mask = _reorient_nifti_mask_to_dicom(seg_path, rtstruct.series_data)
            rtstruct.add_roi(mask=mask, name=name)

        output_path = os.path.join(temp_dir, "rtstruct.dcm")
        rtstruct.save(output_path)

        final_path = os.path.join(
            tempfile.gettempdir(),
            f"voxtell_rtstruct_{os.urandom(8).hex()}.dcm",
        )
        shutil.copy(output_path, final_path)

        logger.info("RTSTRUCT exported: session=%s, structures=%s", session_id, names)

        return FileResponse(
            final_path,
            media_type="application/dicom",
            filename="rtstruct.dcm",
            background=None,
        )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    cleanup_session(session_id)
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "1711"))
    uvicorn.run(app, host="0.0.0.0", port=port)
