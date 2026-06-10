import { X } from 'lucide-react'
import { patientId, type PatientEntry } from '../lib/patientStore'

interface Props {
  patients: PatientEntry[]
  activeId: string | null
  onSelect: (id: string) => void
  onRemove: (id: string) => void
}

export default function PatientTabs({ patients, activeId, onSelect, onRemove }: Props) {
  if (patients.length === 0) return null

  return (
    <div className="flex items-center gap-1.5 ml-4 overflow-x-auto no-scrollbar">
      {patients.map(p => {
        const id = patientId(p.pid, p.series_uid)
        const active = id === activeId
        return (
          <button
            key={id}
            onClick={() => onSelect(id)}
            className={`group flex items-center gap-1.5 pl-3 pr-1.5 py-1 rounded-full text-xs font-medium whitespace-nowrap transition-all ${
              active
                ? 'bg-blue-500/15 text-blue-400 border border-blue-500/30'
                : 'bg-white/5 text-gray-400 border border-transparent hover:bg-white/10 hover:text-white'
            }`}
          >
            Patient {p.pid}
            <span
              onClick={e => { e.stopPropagation(); onRemove(id) }}
              className="w-4 h-4 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 hover:bg-white/10 transition-opacity"
            >
              <X size={10} />
            </span>
          </button>
        )
      })}
    </div>
  )
}
