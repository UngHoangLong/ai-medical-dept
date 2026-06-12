import { useRef, useState } from 'react'
import { X, Upload, FileArchive, FileJson, Loader2, User, CheckCircle2, AlertCircle } from 'lucide-react'

interface PatientPreview {
  pid: string
  series_uid: string
  age?: number
  gender?: string
  smoking?: string
  pkyr?: number
  diseases: string[]
}

interface Props {
  onAnalyze: (fd: FormData) => void
  onClose: () => void
}

function buildPreview(json: Record<string, unknown>): PatientPreview {
  const demo    = (json.demo     as Record<string, unknown>) ?? {}
  const smoking = (json.smoking  as Record<string, unknown>) ?? {}
  const dis     = (json.disease_his as Record<string, unknown>) ?? {}
  const cancer  = (json.cancer_his  as Record<string, unknown>) ?? {}

  return {
    pid:        String(json.pid ?? ''),
    series_uid: String(json.series_uid ?? ''),
    age:        typeof demo.age    === 'number' ? demo.age    : undefined,
    gender:     typeof demo.gender === 'string' ? demo.gender : undefined,
    smoking:    typeof smoking.cigsmok === 'string' ? smoking.cigsmok : undefined,
    pkyr:       typeof smoking.pkyr === 'number' ? smoking.pkyr : undefined,
    diseases:   [...Object.keys(dis), ...Object.keys(cancer)],
  }
}

export default function PatientModal({ onAnalyze, onClose }: Props) {
  const jsonRef = useRef<HTMLInputElement>(null)
  const zipRef  = useRef<HTMLInputElement>(null)

  const [parsedJson, setParsedJson] = useState<Record<string, unknown> | null>(null)
  const [preview,    setPreview]    = useState<PatientPreview | null>(null)
  const [jsonError,  setJsonError]  = useState<string | null>(null)
  const [zipFile,    setZipFile]    = useState<File | null>(null)
  const [submitting, setSubmitting] = useState(false)

  function handleJsonFile(file: File) {
    setJsonError(null)
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target?.result as string) as Record<string, unknown>
        if (!json.pid)        throw new Error('Missing "pid" field')
        if (!json.series_uid) throw new Error('Missing "series_uid" field')
        setParsedJson(json)
        setPreview(buildPreview(json))
      } catch (err) {
        setJsonError(err instanceof Error ? err.message : 'Invalid JSON file')
        setParsedJson(null)
        setPreview(null)
      }
    }
    reader.readAsText(file)
  }

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    if (!parsedJson || !zipFile) return
    setSubmitting(true)

    const { pid, series_uid, ...clinical_data } = parsedJson

    const fd = new FormData()
    fd.append('pid',           String(pid))
    fd.append('series_uid',    String(series_uid))
    fd.append('clinical_data', JSON.stringify(clinical_data))
    fd.append('dicom_zip',     zipFile, zipFile.name)

    onAnalyze(fd)
  }

  const canSubmit = parsedJson !== null && zipFile !== null && !submitting

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />

      {/* Card */}
      <div className="relative w-full max-w-md bg-gray-900 border border-white/10 rounded-3xl shadow-2xl shadow-black/50 overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-6 pb-4">
          <div>
            <h2 className="text-lg font-semibold text-white">New Patient</h2>
            <p className="text-xs text-gray-500 mt-0.5">Upload patient info and CT scan</p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-xl flex items-center justify-center text-gray-500 hover:text-white hover:bg-white/10 transition-all"
          >
            <X size={16} />
          </button>
        </div>

        <div className="h-px bg-white/5 mx-6" />

        <form onSubmit={handleSubmit} className="px-6 py-5 space-y-4">

          {/* ── Patient Info JSON ─────────────────────────────────── */}
          <div>
            <label className="block text-xs font-medium text-gray-400 mb-1.5">
              Patient Info
              <span className="ml-1.5 text-gray-600 font-normal">.json</span>
            </label>

            <div
              onClick={() => jsonRef.current?.click()}
              className={`rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-200 ${
                preview
                  ? 'border-green-500/30 bg-green-500/5 p-4'
                  : jsonError
                  ? 'border-red-500/30 bg-red-500/5 p-5 text-center'
                  : 'border-white/10 hover:border-white/20 p-5 text-center'
              }`}
            >
              {preview ? (
                /* ── Patient preview card ── */
                <div className="flex items-start gap-3">
                  <div className="w-9 h-9 rounded-xl bg-blue-500/15 border border-blue-500/20 flex items-center justify-center shrink-0 mt-0.5">
                    <User size={16} className="text-blue-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-semibold text-white">
                        Patient {preview.pid}
                      </p>
                      <CheckCircle2 size={14} className="text-green-400 shrink-0" />
                    </div>
                    <p className="text-xs text-gray-400 mt-0.5">
                      {[
                        preview.gender,
                        preview.age ? `${preview.age} y/o` : null,
                        preview.smoking ? `${preview.smoking} smoker` : null,
                        preview.pkyr ? `${preview.pkyr} pack-yr` : null,
                      ].filter(Boolean).join(' · ')}
                    </p>
                    {preview.diseases.length > 0 && (
                      <p className="text-xs text-gray-500 mt-0.5 truncate">
                        {preview.diseases.join(', ')}
                      </p>
                    )}
                    <p className="text-xs text-gray-600 mt-1 truncate">
                      {preview.series_uid}
                    </p>
                  </div>
                  <span className="text-xs text-gray-600 hover:text-gray-400 shrink-0 mt-0.5">
                    Change
                  </span>
                </div>
              ) : jsonError ? (
                <div className="flex flex-col items-center gap-2">
                  <AlertCircle size={22} className="text-red-400" />
                  <p className="text-sm text-red-400">{jsonError}</p>
                  <p className="text-xs text-gray-600">Click to try another file</p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2">
                  <FileJson size={22} className="text-gray-600" />
                  <p className="text-sm text-gray-500">Click to upload</p>
                  <p className="text-xs text-gray-600">
                    JSON with pid · series_uid · clinical data
                  </p>
                </div>
              )}
            </div>
            <input
              ref={jsonRef}
              type="file"
              accept=".json"
              className="hidden"
              onChange={e => {
                const f = e.target.files?.[0]
                if (f) handleJsonFile(f)
              }}
            />
          </div>

          {/* ── DICOM ZIP ─────────────────────────────────────────── */}
          <div>
            <label className="block text-xs font-medium text-gray-400 mb-1.5">
              DICOM Archive
              <span className="ml-1.5 text-gray-600 font-normal">.zip</span>
            </label>

            <div
              onClick={() => zipRef.current?.click()}
              className={`rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-200 ${
                zipFile
                  ? 'border-blue-500/30 bg-blue-500/5 p-4'
                  : 'border-white/10 hover:border-white/20 p-5 text-center'
              }`}
            >
              {zipFile ? (
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-blue-500/15 border border-blue-500/20 flex items-center justify-center shrink-0">
                    <FileArchive size={16} className="text-blue-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">{zipFile.name}</p>
                    <p className="text-xs text-gray-500">{(zipFile.size / 1e6).toFixed(1)} MB</p>
                  </div>
                  <CheckCircle2 size={14} className="text-green-400 shrink-0" />
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2">
                  <Upload size={22} className="text-gray-600" />
                  <p className="text-sm text-gray-500">Click to upload</p>
                  <p className="text-xs text-gray-600">ZIP archive of DICOM files</p>
                </div>
              )}
            </div>
            <input
              ref={zipRef}
              type="file"
              accept=".zip"
              className="hidden"
              onChange={e => setZipFile(e.target.files?.[0] ?? null)}
            />
          </div>

          {/* ── Actions ───────────────────────────────────────────── */}
          <div className="flex gap-3 pt-1">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-2.5 rounded-xl border border-white/10 text-gray-400 hover:text-white hover:bg-white/5 text-sm font-medium transition-all"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!canSubmit}
              className="flex-1 py-2.5 rounded-xl bg-blue-500 hover:bg-blue-400 disabled:bg-white/5 disabled:text-gray-600 text-white text-sm font-medium transition-all flex items-center justify-center gap-2 shadow-lg shadow-blue-500/20 disabled:shadow-none"
            >
              {submitting ? (
                <><Loader2 size={14} className="animate-spin" /> Starting...</>
              ) : 'Run Analysis'}
            </button>
          </div>

        </form>
      </div>
    </div>
  )
}
