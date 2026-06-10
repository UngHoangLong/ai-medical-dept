from pydantic import BaseModel


class RadiologyStepResult(BaseModel):
    answer: dict[str, str]


class RadiologyResult(BaseModel):
    screening: RadiologyStepResult
    detail: RadiologyStepResult | None = None


class CardiologyResult(BaseModel):
    # {"CVD_diagnosis": "No", "CVD_mortality": "Low risk"}
    answer: dict[str, str]


class OncologyResult(BaseModel):
    # {"lung_cancer_risk": "No cancer within follow-up"}
    answer: dict[str, str]


class FindingImpressionResult(BaseModel):
    findings: str | None = None
    impression: str | None = None


class VerificationResult(BaseModel):
    # free-text narrative từ MedGemma base (thought block + bất kỳ text nào sau đó)
    analysis: str | None = None


class AnalyzeResponse(BaseModel):
    radiology: RadiologyResult
    cardiology: CardiologyResult
    oncology: OncologyResult
    finding_impression: FindingImpressionResult
    verification: VerificationResult
