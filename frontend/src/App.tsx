import { useState, useEffect, useRef } from 'react'
import { UserPlus, MessageSquare, MessageSquareOff, History, FileText } from 'lucide-react'
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
  const [reportOpen, setReportOpen] = useState(true)
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Multi-patient session — persisted in sessionStorage, restored on F5
  const [patients, setPatients] = useState<PatientEntry[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [reportId, setReportId] = useState<string | null>(() => loadActiveReportId())

  // Horizontal Panel Resizing States
  const [ctWidth, setCtWidth] = useState<number>(() => {
    const saved = localStorage.getItem('medai_ct_width')
    return saved ? parseInt(saved, 10) : 450
  })
  const [chatWidth, setChatWidth] = useState<number>(() => {
    const saved = localStorage.getItem('medai_chat_width')
    return saved ? parseInt(saved, 10) : 300
  })

  const ctWidthRef = useRef(ctWidth)
  const chatWidthRef = useRef(chatWidth)
  const chatOpenRef = useRef(chatOpen)
  const reportOpenRef = useRef(reportOpen)
  const isDraggingCt = useRef(false)
  const isDraggingChat = useRef(false)

  useEffect(() => {
    ctWidthRef.current = ctWidth
  }, [ctWidth])

  useEffect(() => {
    chatWidthRef.current = chatWidth
  }, [chatWidth])

  useEffect(() => {
    chatOpenRef.current = chatOpen
  }, [chatOpen])

  useEffect(() => {
    reportOpenRef.current = reportOpen
  }, [reportOpen])

  const handleMouseDownCt = (e: React.MouseEvent) => {
    e.preventDefault()
    isDraggingCt.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  const handleMouseDownChat = (e: React.MouseEvent) => {
    e.preventDefault()
    isDraggingChat.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDraggingCt.current) {
        const container = document.getElementById('panels-container')
        if (container) {
          const rect = container.getBoundingClientRect()
          const newWidth = e.clientX - rect.left
          const currentChatWidth = chatOpenRef.current ? chatWidthRef.current : 0
          const clampedWidth = Math.max(380, Math.min(newWidth, rect.width - currentChatWidth - 300))
          setCtWidth(clampedWidth)
        }
      } else if (isDraggingChat.current) {
        const container = document.getElementById('panels-container')
        if (container) {
          const rect = container.getBoundingClientRect()
          const newWidth = rect.right - e.clientX
          const currentCtWidth = reportOpenRef.current ? ctWidthRef.current : 380
          const minMiddleWidth = reportOpenRef.current ? 300 : 0
          const clampedWidth = Math.max(260, Math.min(newWidth, rect.width - currentCtWidth - minMiddleWidth))
          setChatWidth(clampedWidth)
        }
      }
    }

    const handleMouseUp = () => {
      if (isDraggingCt.current || isDraggingChat.current) {
        isDraggingCt.current = false
        isDraggingChat.current = false
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
        localStorage.setItem('medai_ct_width', String(ctWidthRef.current))
        localStorage.setItem('medai_chat_width', String(chatWidthRef.current))
        window.dispatchEvent(new Event('resize'))
      }
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [])


  useEffect(() => {
    const restored = loadPatients()
    if (restored.length > 0) {
      setPatients(restored)
      const active = loadActiveId()
      const fallback = patientId(restored[0].pid, restored[0].series_uid)
      const nextActiveId = active && restored.some(p => patientId(p.pid, p.series_uid) === active) ? active : fallback
      setActiveId(nextActiveId)

      const restoredReportId = restored.find(p => patientId(p.pid, p.series_uid) === nextActiveId)?.result?.report_id ?? null
      setReportId(restoredReportId)
      saveActiveReportId(restoredReportId)

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

          {/* Report toggle */}
          <button
            onClick={() => setReportOpen(o => !o)}
            title={reportOpen ? 'Hide report' : 'Show report'}
            className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all ${
              reportOpen ? 'text-blue-400 bg-blue-500/5' : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`}
          >
            <FileText size={16} />
          </button>

          {/* Chat toggle */}
          <button
            onClick={() => setChatOpen(o => !o)}
            title={chatOpen ? 'Hide chat' : 'Show chat'}
            className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all ${
              chatOpen ? 'text-blue-400 bg-blue-500/5' : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`}
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
      <div id="panels-container" className="flex flex-1 overflow-hidden p-3">

        {/* Left — CT Viewer */}
        <CTViewer
          status={status}
          pid={active?.pid}
          seriesUid={active?.series_uid}
          style={reportOpen ? { width: `${ctWidth}px` } : undefined}
          className={reportOpen ? "shrink-0" : "flex-1"}
        />

        {/* Drag handle for CT Viewer */}
        {reportOpen && (
          <div
            onMouseDown={handleMouseDownCt}
            className="w-3 shrink-0 cursor-col-resize flex items-center justify-center group select-none"
          >
            <div className="w-[2px] h-10 bg-slate-800 group-hover:bg-blue-500 group-active:bg-blue-400 rounded transition-colors duration-150" />
          </div>
        )}

        {/* Center — Agent Results */}
        {reportOpen && (
          <AgentResults
            result={result}
            status={status}
            errorMsg={errorMsg}
          />
        )}

        {/* Right — Chat */}
        {chatOpen && (
          <>
            {/* Drag handle for Chat Panel */}
            <div
              onMouseDown={handleMouseDownChat}
              className="w-3 shrink-0 cursor-col-resize flex items-center justify-center group select-none"
            >
              <div className="w-[2px] h-10 bg-slate-800 group-hover:bg-blue-500 group-active:bg-blue-400 rounded transition-colors duration-150" />
            </div>

            <ChatPanel
              result={result}
              reportId={reportId ?? result?.report_id ?? null}
              cacheKey={cacheKey}
              status={status}
              style={{ width: `${chatWidth}px` }}
              className="shrink-0"
            />
          </>
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
