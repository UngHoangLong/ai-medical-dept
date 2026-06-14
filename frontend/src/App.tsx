import { useState, useEffect, useRef } from 'react'
import { UserPlus, MessageSquare, MessageSquareOff, History } from 'lucide-react'
import type { AnalyzeResponse } from './types/api'
import CTViewer from './components/CTViewer'
import AgentResults from './components/AgentResults'
import ChatPanel from './components/ChatPanel'
import PatientModal from './components/PatientModal'
import PatientHistoryModal from './components/PatientHistoryModal'
import PatientTabs from './components/PatientTabs'
import {
  type PatientEntry, patientId,
  loadPatients, savePatients, loadActiveId, saveActiveId,
  loadActiveReportId, saveActiveReportId,
} from './lib/patientStore'

export type AnalysisStatus = 'idle' | 'loading' | 'done' | 'error'

const BACKEND = import.meta.env.VITE_BACKEND_URL ?? ''

export default function App() {
  const [status, setStatus] = useState<AnalysisStatus>('idle')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [historyLoadingId, setHistoryLoadingId] = useState<string | null>(null)
  const [chatOpen, setChatOpen] = useState(true)
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Multi-patient session — persisted in sessionStorage, restored on F5
  const [patients, setPatients] = useState<PatientEntry[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [reportId, setReportId] = useState<string | null>(() => loadActiveReportId())


  useEffect(() => {
    const restored = loadPatients()
    if (restored.length > 0) {
      setPatients(restored)
      const active = loadActiveId()
      const fallback = patientId(restored[0].pid, restored[0].series_uid)
      const nextActiveId = active && restored.some(p => patientId(p.pid, p.series_uid) === active) ? active : fallback
      setActiveId(nextActiveId)

      const savedReportId = loadActiveReportId()
      const restoredReportId = restored.find(p => patientId(p.pid, p.series_uid) === nextActiveId)?.result?.report_id ?? null
      const finalReportId = savedReportId ?? restoredReportId
      if (finalReportId) {
        setReportId(finalReportId)
        saveActiveReportId(finalReportId)
      }

      setStatus('done')
    }
  }, [])

  const active = patients.find(p => patientId(p.pid, p.series_uid) === activeId) ?? null
  const result = active?.result ?? null
  const cacheKey = active ? patientId(active.pid, active.series_uid) : ''

  useEffect(() => {
    if (status === 'loading') {
      setElapsed(0)
      timerRef.current = setInterval(() => setElapsed(s => s + 1), 1000)
    } else {
      if (timerRef.current) clearInterval(timerRef.current)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [status])

  function formatElapsed(s: number) {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`
  }

  async function handleAnalyze(formData: FormData) {
    setModalOpen(false)
    setStatus('loading')
    setErrorMsg(null)
    try {
      const resp = await fetch(`${BACKEND}/api/v1/analyze`, {
        method: 'POST',
        body: formData,
      })
      if (!resp.ok) {
        const text = await resp.text()
        throw new Error(`HTTP ${resp.status}: ${text.slice(0, 200)}`)
      }
      const data: AnalyzeResponse = await resp.json()
      const fetched_id = data.report_id ?? 'unknown_report_id'
      setReportId(fetched_id)
      saveActiveReportId(fetched_id)


      const pid = String(formData.get('pid'))
      const series_uid = String(formData.get('series_uid'))
      const id = patientId(pid, series_uid)

      setPatients(prev => {
        const next = [...prev.filter(p => patientId(p.pid, p.series_uid) !== id), { pid, series_uid, result: data }]
        savePatients(next)
        return next
      })
      setActiveId(id)
      saveActiveId(id)
      setStatus('done')
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : 'Unknown error')
      setStatus('error')
    }
  }

  function handleSelectPatient(id: string) {
    setActiveId(id)
    saveActiveId(id)

    const selected = patients.find(p => patientId(p.pid, p.series_uid) === id)
    const nextReportId = selected?.result?.report_id ?? null
    setReportId(nextReportId)
    saveActiveReportId(nextReportId)

    setErrorMsg(null)
    setStatus('done')
  }

  async function handleLoadFromHistory(pid: string, series_uid: string) {
    const id = patientId(pid, series_uid)

    // Đã có sẵn trong session — chỉ cần chuyển tab
    if (patients.some(p => patientId(p.pid, p.series_uid) === id)) {
      setHistoryOpen(false)
      handleSelectPatient(id)
      return
    }

    setHistoryLoadingId(id)
    try {
      const resp = await fetch(`${BACKEND}/api/v1/analysis/${pid}/${series_uid}`)
      if (!resp.ok) {
        const text = await resp.text()
        throw new Error(`HTTP ${resp.status}: ${text.slice(0, 200)}`)
      }
      const data: AnalyzeResponse = await resp.json()

      setPatients(prev => {
        const next = [...prev.filter(p => patientId(p.pid, p.series_uid) !== id), { pid, series_uid, result: data }]
        savePatients(next)
        return next
      })
      setActiveId(id)
      saveActiveId(id)
      const nextReportId = data.report_id ?? null
      setReportId(nextReportId)
      saveActiveReportId(nextReportId)
      setErrorMsg(null)
      setStatus('done')
      setHistoryOpen(false)
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setHistoryLoadingId(null)
    }
  }

  function handleRemovePatient(id: string) {
    setPatients(prev => {
      const next = prev.filter(p => patientId(p.pid, p.series_uid) !== id)
      savePatients(next)
      if (activeId === id) {
        const fallback = next.length > 0 ? patientId(next[0].pid, next[0].series_uid) : null
        const fallbackReportId = next.length > 0 ? next[0].result?.report_id ?? null : null
        setActiveId(fallback)
        saveActiveId(fallback)
        setReportId(fallbackReportId)
        saveActiveReportId(fallbackReportId)
        setStatus(fallback ? 'done' : 'idle')
      }
      return next
    })
  }

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-white overflow-hidden">

      {/* ── Header ────────────────────────────────────────────────── */}
      <header className="h-14 shrink-0 flex items-center px-5 border-b border-white/5 bg-gray-900/80 backdrop-blur-sm">
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-500 to-violet-500 flex items-center justify-center font-bold text-xs shadow-lg shadow-blue-500/20">
            M
          </div>
          <div>
            <p className="text-sm font-semibold leading-none">MedAI</p>
            <p className="text-xs text-gray-500 leading-none mt-0.5">Lung Cancer Screening</p>
          </div>
        </div>

        {/* Status pill */}
        {status === 'loading' && (
          <div className="ml-6 flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
            <span className="text-xs text-blue-400">
              Analyzing... {formatElapsed(elapsed)}
            </span>
            {elapsed < 30 && (
              <span className="text-xs text-blue-400/50">· cold start may take ~6 min</span>
            )}
          </div>
        )}
        {status === 'error' && (
          <div className="ml-6 flex items-center gap-2 px-3 py-1 rounded-full bg-red-500/10 border border-red-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
            <span className="text-xs text-red-400">Analysis failed</span>
          </div>
        )}

        {/* Patient tabs — switch between analyzed patients in this session */}
        <PatientTabs
          patients={patients}
          activeId={activeId}
          onSelect={handleSelectPatient}
          onRemove={handleRemovePatient}
        />

        <div className="ml-auto flex items-center gap-2">
          {/* Patient history */}
          <button
            onClick={() => setHistoryOpen(true)}
            title="Patient history"
            className="w-9 h-9 rounded-xl flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/5 transition-all"
          >
            <History size={16} />
          </button>

          {/* Chat toggle */}
          <button
            onClick={() => setChatOpen(o => !o)}
            title={chatOpen ? 'Hide chat' : 'Show chat'}
            className="w-9 h-9 rounded-xl flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/5 transition-all"
          >
            {chatOpen ? <MessageSquareOff size={16} /> : <MessageSquare size={16} />}
          </button>

          {/* Add Patient */}
          <button
            onClick={() => setModalOpen(true)}
            className="flex items-center gap-2 bg-blue-500 hover:bg-blue-400 text-white text-sm font-medium px-4 py-2 rounded-xl transition-all duration-200 shadow-lg shadow-blue-500/20 hover:shadow-blue-400/30"
          >
            <UserPlus size={14} />
            Add Patient
          </button>
        </div>
      </header>

      {/* ── 3-panel body ──────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden gap-3 p-3">

        {/* Left — CT Viewer */}
        <CTViewer status={status} />

        {/* Center — Agent Results */}
        <AgentResults
          result={result}
          status={status}
          errorMsg={errorMsg}
        />

        {/* Right — Chat */}
        {chatOpen && (
          <ChatPanel
            result={result}
            reportId={reportId ?? result?.report_id ?? null}
            cacheKey={cacheKey}
            status={status}
          />
        )}
      </div>

      {/* ── Modal ─────────────────────────────────────────────────── */}
      {modalOpen && (
        <PatientModal
          onAnalyze={handleAnalyze}
          onClose={() => setModalOpen(false)}
        />
      )}

      {historyOpen && (
        <PatientHistoryModal
          onSelect={handleLoadFromHistory}
          onClose={() => setHistoryOpen(false)}
          loadingId={historyLoadingId}
        />
      )}
    </div>
  )
}
