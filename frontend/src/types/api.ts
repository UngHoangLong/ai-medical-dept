export interface RadiologyScreeningAnswer {
  chest_abn_54: string | null  // atelectasis
  chest_abn_55: string | null  // pleural effusion
  chest_abn_56: string | null  // hilar/mediastinal mass
  chest_abn_57: string | null  // chest wall abnormality
  chest_abn_58: string | null  // consolidation
  chest_abn_59: string | null  // emphysema
  chest_abn_61: string | null  // fibrosis/honeycombing
  nodule_presence: string | null
}

export interface RadiologyDetailAnswer {
  nodule_location: string | null
  nodule_attenuation: string | null
  nodule_margin: string | null
  nodule_size: string | null
}

export interface CardiologyAnswer {
  CVD_diagnosis: string | null
  CVD_mortality: string | null
}

export interface OncologyAnswer {
  lung_cancer_risk: string | null
}

export interface AnalyzeResponse {
  report_id: string | null
  radiology: {
    screening: { answer: RadiologyScreeningAnswer }
    detail: { answer: RadiologyDetailAnswer } | null
  }
  cardiology: { answer: CardiologyAnswer }
  oncology: { answer: OncologyAnswer }
  finding_impression: {
    findings: string | null
    impression: string | null
  }
  verification: {
    analysis: string | null
  }
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface AnalysisListItem {
  pid: string
  series_uid: string
  last_modified: string
}
