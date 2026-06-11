import { useEffect, useState } from 'react'
import { X, History, User, Loader2, AlertCircle } from 'lucide-react'
import type { AnalysisListItem } from '../types/api'

const BACKEND = import.meta.env.VITE_BACKEND_URL ?? ''

interface Props {
  onSelect: (pid: string, series_uid: string) => void
  onClose: () => void
  loadingId: string | null
}

export default function PatientHistoryModal({ onSelect, onClose, loadingId }: Props) {
  const [items, setItems] = useState<AnalysisListItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const resp = await fetch(`${BACKEND}/api/v1/analyses`)
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const data: AnalysisListItem[] = await resp.json()
        if (!cancelled) setItems(data)
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />

      {/* Card */}
      <div className="relative w-full max-w-md bg-gray-900 border border-white/10 rounded-3xl shadow-2xl shadow-black/50 overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-6 pb-4">
          <div>
            <h2 className="text-lg font-semibold text-white">Patient History</h2>
            <p className="text-xs text-gray-500 mt-0.5">Previously analyzed patients</p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-xl flex items-center justify-center text-gray-500 hover:text-white hover:bg-white/10 transition-all"
          >
            <X size={16} />
          </button>
        </div>

        <div className="h-px bg-white/5 mx-6" />

        <div className="px-6 py-5 max-h-[60vh] overflow-y-auto">
          {loading && (
            <div className="flex items-center justify-center gap-2 py-8 text-gray-500 text-sm">
              <Loader2 size={16} className="animate-spin" /> Loading...
            </div>
          )}

          {error && (
            <div className="flex flex-col items-center gap-2 py-8 text-center">
              <AlertCircle size={22} className="text-red-400" />
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          {!loading && !error && items.length === 0 && (
            <div className="flex flex-col items-center gap-2 py-8 text-center">
              <History size={22} className="text-gray-600" />
              <p className="text-sm text-gray-500">No analyzed patients yet</p>
            </div>
          )}

          {!loading && !error && items.length > 0 && (
            <div className="space-y-1.5">
              {items.map(item => {
                const id = `${item.pid}/${item.series_uid}`
                const isLoading = loadingId === id
                return (
                  <button
                    key={id}
                    disabled={loadingId !== null}
                    onClick={() => onSelect(item.pid, item.series_uid)}
                    className="w-full flex items-center gap-3 p-3 rounded-2xl border border-white/5 hover:border-white/10 hover:bg-white/5 transition-all text-left disabled:opacity-50"
                  >
                    <div className="w-9 h-9 rounded-xl bg-blue-500/15 border border-blue-500/20 flex items-center justify-center shrink-0">
                      <User size={16} className="text-blue-400" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold text-white">Patient {item.pid}</p>
                      <p className="text-xs text-gray-600 mt-0.5 truncate">{item.series_uid}</p>
                    </div>
                    {isLoading && <Loader2 size={14} className="animate-spin text-blue-400 shrink-0" />}
                  </button>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
