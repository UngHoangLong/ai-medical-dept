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
  // --- PHẦN 3: XỬ LÝ CHUỖI INPUT RỖNG / MIC LỖI (CHẶN NGAY TẠI FRONTEND) ---
  // Nếu file âm thanh quá nhỏ (dưới 4KB), chứng tỏ không có dữ liệu âm thanh thực tế
  if (!audioBlob || audioBlob.size < 4000) {
    alert('Không phát hiện thấy dữ liệu giọng nói. Vui lòng kiểm tra lại thiết bị Micro của bạn.')
    return ''
  }

  // --- PHẦN 1: ĐỔI SANG ĐỊNH DẠNG CHÍNH XÁC CỦA TRÌNH DUYỆT ---
  // Lấy mimeType thực tế của blob (ví dụ: audio/webm hoặc audio/mp4), thay vì ép cứng .wav
  const mimeType = audioBlob.type || 'audio/webm'
  // Trích xuất đuôi mở rộng từ mimeType để đặt tên file cho đúng (ví dụ: audio/webm -> webm)
  const fileExtension = mimeType.split(';')[0].split('/')[1] || 'webm'

  const formData = new FormData()
  formData.append('file', audioBlob, `audio.${fileExtension}`)

  try {
    const response = await axios.post(`${BACKEND}/api/v1/stt`, formData)

    // Nếu backend trả về thành công nhưng chuỗi text rỗng (Whisper không nghe thấy gì)
    if (!response.data.text || response.data.text.trim() === '') {
      alert('Không thể nhận diện được chữ viết. Hãy thử nói rõ ràng hoặc gần micro hơn.')
      return ''
    }

    return response.data.text
  } catch (error: any) {
    // --- PHẦN 2: THÊM THÔNG BÁO KHI GẶP TRỤC TRẶC KỸ THUẬT / LỖI MẠNG ---
    console.error('Failed to transcribe audio:', error)

    if (error.response) {
      // Server trả về code lỗi (Ví dụ: 422, 500)
      alert(`Lỗi hệ thống STT (${error.response.status}): Máy chủ xử lý âm thanh thất bại.`)
    } else if (error.request) {
      // Gửi request đi nhưng không nhận được phản hồi (Timeout / Mất mạng)
      alert('Không thể kết nối đến máy chủ STT. Vui lòng kiểm tra lại đường truyền mạng.')
    } else {
      // Các lỗi cấu hình khác
      alert('Đã xảy ra lỗi không xác định trong quá trình xử lý micro.')
    }

    return ''
  }
}