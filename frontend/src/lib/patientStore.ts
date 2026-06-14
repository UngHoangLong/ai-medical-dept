import axios from 'axios'
import type { AnalyzeResponse } from '../types/api'

const BACKEND = import.meta.env.VITE_BACKEND_URL ?? ''

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

export async function transcribeAudio(audioBlob: Blob): Promise<string> {
  const formData = new FormData()
  formData.append('file', audioBlob, 'audio.wav')

  try {
    const response = await axios.post(`${BACKEND}/api/v1/stt`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data.text || ''
  } catch (error) {
    console.error('Failed to transcribe audio:', error)
    return ''
  }
}
