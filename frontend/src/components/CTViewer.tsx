import { ScanLine, Layers } from 'lucide-react'
import type { AnalysisStatus } from '../App'

interface Props {
  status: AnalysisStatus
}

export default function CTViewer({ status }: Props) {
  return (
    <aside className="w-64 shrink-0 bg-black rounded-2xl border border-white/5 flex flex-col overflow-hidden">
      {/* Panel header */}
      <div className="px-4 py-3 border-b border-white/5 flex items-center gap-2">
        <ScanLine size={14} className="text-gray-500" />
        <span className="text-xs font-medium text-gray-400">CT Viewer</span>
      </div>

      {/* Content */}
      <div className="flex-1 flex flex-col items-center justify-center p-4">
        {status === 'loading' ? (
          <div className="text-center">
            <div className="w-12 h-12 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mx-auto mb-3">
              <ScanLine size={22} className="text-blue-400 animate-pulse" />
            </div>
            <p className="text-xs text-gray-500">Loading scan...</p>
          </div>
        ) : status === 'done' ? (
          <div className="text-center w-full">
            {/* Placeholder CT image area */}
            <div className="w-full aspect-square rounded-xl bg-gray-950 border border-white/5 flex flex-col items-center justify-center mb-3">
              <Layers size={28} className="text-gray-700 mb-2" />
              <p className="text-xs text-gray-600">Viewer</p>
              <p className="text-xs text-gray-700">coming soon</p>
            </div>
            <div className="flex items-center justify-between text-xs text-gray-600 px-1">
              <span>◄</span>
              <span>— / 85</span>
              <span>►</span>
            </div>
          </div>
        ) : (
          <div className="text-center">
            <div className="w-12 h-12 rounded-2xl bg-white/3 border border-white/8 flex items-center justify-center mx-auto mb-3">
              <ScanLine size={22} className="text-gray-700" />
            </div>
            <p className="text-xs text-gray-600 leading-relaxed">
              No scan loaded
            </p>
          </div>
        )}
      </div>
    </aside>
  )
}
