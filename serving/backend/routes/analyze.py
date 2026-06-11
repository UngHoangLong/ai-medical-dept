import json

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from serving.backend.schemas import AnalysisListItem, AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    pid: str = Form(...),
    series_uid: str = Form(...),
    clinical_data: str = Form(...),   # JSON string
    dicom_zip: UploadFile = File(..., description="1 file .zip chứa toàn bộ .dcm của 1 CT series"),
):
    try:
        clinical = json.loads(clinical_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="clinical_data is not valid JSON")

    if not dicom_zip.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=422, detail="dicom_zip must be a .zip file containing .dcm slices")

    dicom_zip_bytes = await dicom_zip.read()

    pipeline = request.app.state.pipeline
    try:
        result = await pipeline.run(pid, series_uid, clinical, dicom_zip_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lấy report đã phân tích trước đó.
@router.get("/analysis/{pid}/{series_uid}", response_model=AnalyzeResponse)
async def get_analysis(request: Request, pid: str, series_uid: str):
    pipeline = request.app.state.pipeline
    result = await pipeline.get_analysis(pid, series_uid)
    if result is None:
        raise HTTPException(status_code=404, detail="No analysis found for this patient/series")
    return result


# Danh sách bệnh nhân đã có kết quả phân tích lưu trên S3.
@router.get("/analyses", response_model=list[AnalysisListItem])
async def list_analyses(request: Request):
    pipeline = request.app.state.pipeline
    return await pipeline.list_analyses()
