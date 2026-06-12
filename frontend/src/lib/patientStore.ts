import type { AnalyzeResponse } from '../types/api'

export interface PatientEntry {
  pid: string
  series_uid: string
  result: AnalyzeResponse
}

const PATIENTS_KEY = 'medai_patients'
const ACTIVE_KEY = 'medai_active_id'

export function patientId(pid: string, series_uid: string): string {
  return `${pid}/${series_uid}`
}

export function loadPatients(): PatientEntry[] {
  try {
    const raw = sessionStorage.getItem(PATIENTS_KEY)
    return raw ? (JSON.parse(raw) as PatientEntry[]) : []
  } catch {
    return []
  }
}

export function savePatients(patients: PatientEntry[]) {
  sessionStorage.setItem(PATIENTS_KEY, JSON.stringify(patients))
}

export function loadActiveId(): string | null {
  return sessionStorage.getItem(ACTIVE_KEY)
}

export function saveActiveId(id: string | null) {
  if (id) sessionStorage.setItem(ACTIVE_KEY, id)
  else sessionStorage.removeItem(ACTIVE_KEY)
}
