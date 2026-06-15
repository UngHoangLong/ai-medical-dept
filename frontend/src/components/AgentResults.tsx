import { useState } from 'react'
import { ChevronDown, ChevronUp, Stethoscope, Heart, FlaskConical, FileText, ShieldCheck, Lock } from 'lucide-react'
import type { AnalyzeResponse } from '../types/api'
import type { AnalysisStatus } from '../App'

const TASK_LABEL: Record<string, string> = {
  chest_abn_54: 'Atelectasis',
  chest_abn_55: 'Pleural Effusion',
  chest_abn_56: 'Hilar / Mediastinal Mass',
  chest_abn_57: 'Chest Wall Abnormality',
  chest_abn_58: 'Consolidation',
  chest_abn_59: 'Emphysema',
  chest_abn_61: 'Fibrosis / Honeycombing',
  nodule_presence: 'Nodule Present',
  nodule_location: 'Location',
  nodule_attenuation: 'Attenuation',
  nodule_margin: 'Margin',
  nodule_size: 'Size',
  CVD_diagnosis: 'CVD Diagnosis',
  CVD_mortality: 'CVD Mortality Risk',
  lung_cancer_risk: 'Lung Cancer Risk',
}

// Câu hỏi gốc đã đưa cho model — hiện dạng tooltip để bác sĩ hiểu ngữ cảnh
const TASK_QUESTION: Record<string, string> = {
  chest_abn_54: 'Is there any atelectasis, segmental or greater?',
  chest_abn_55: 'Is there any pleural thickening or effusion?',
  chest_abn_56: 'Is there a non-calcified mass or adenopathy ≥10mm in the hilar/mediastinal region?',
  chest_abn_57: 'Is there any chest wall abnormality (bone destruction or metastasis)?',
  chest_abn_58: 'Is there any consolidation?',
  chest_abn_59: 'Is there any emphysema?',
  chest_abn_61: 'Is there any fibrosis, honeycombing, reticular/reticulonodular opacities, or scarring?',
  nodule_presence: 'Is there any lung nodule?',
  nodule_location: 'In which lobe is the nodule located?',
  nodule_attenuation: 'What is the predominant attenuation of the nodule?',
  nodule_margin: 'What is the nodule margin type?',
  nodule_size: 'What is the nodule size?',
  CVD_diagnosis: 'Can you identify any notable abnormalities in the cardiovascular system?',
  CVD_mortality: 'Predict the risk of cardiovascular disease mortality.',
  lung_cancer_risk: "Given the current indicators, what's the six-year outlook for developing lung cancer?",
}

// ── Render free-text đã được annotate bằng <hl c="critical|warning|normal"> tags ──
// Tags này do backend (OpenAI, text-only) chèn vào — đã validate không làm
// thay đổi nội dung gốc. Parse ở đây chỉ để render màu, không tự suy luận thêm.
const HL_CLASS: Record<string, string> = {
  critical: 'bg-red-500/20 text-red-300',
  warning:  'bg-amber-500/20 text-amber-300',
  normal:   'bg-green-500/20 text-green-300',
}

const HL_RE = /<hl c="(critical|warning|normal)">(.*?)<\/hl>/gs

// ── Render markdown nhẹ (**bold**, *italic*) — model verify đôi khi tự chèn
// markdown để nhấn mạnh section (vd "**Chest:**", "*Other*:") ──
const MD_RE = /\*\*(.+?)\*\*|\*(.+?)\*/g

function renderMarkdown(text: string, keyPrefix: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = []
  let cursor = 0
  let i = 0

  for (const m of text.matchAll(MD_RE)) {
    const start = m.index ?? 0
    if (start > cursor) nodes.push(text.slice(cursor, start))
    if (m[1] !== undefined) {
      nodes.push(<strong key={`${keyPrefix}-${i++}`}>{m[1]}</strong>)
    } else {
      nodes.push(<em key={`${keyPrefix}-${i++}`}>{m[2]}</em>)
    }
    cursor = start + m[0].length
  }
  if (cursor < text.length) nodes.push(text.slice(cursor))
  return nodes
}

function renderAnnotated(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = []
  let cursor = 0
  let i = 0

  for (const m of text.matchAll(HL_RE)) {
    const start = m.index ?? 0
    if (start > cursor) nodes.push(...renderMarkdown(text.slice(cursor, start), `pre-${i}`))
    nodes.push(
      <mark key={`hl-${i}`} className={`${HL_CLASS[m[1]]} rounded px-0.5`}>
        {renderMarkdown(m[2], `hl-${i}`)}
      </mark>
    )
    i++
    cursor = start + m[0].length
  }
  if (cursor < text.length) nodes.push(...renderMarkdown(text.slice(cursor), `post-${i}`))
  return nodes
}

// ── Verification (Agent 5) thường tự chèn các tiêu đề section dạng
// "FINDINGS:", "IMPRESSION:", "FINDINGS AND IMPRESSION:"... — tách ra thành
// section riêng để hiển thị giống Agent 4, không bắt buộc model phải tuân
// theo 1 cách viết cố định ──
// Chỉ tách khi cả dòng là CHỮ HOA + dấu ":" đứng riêng (làm tiêu đề riêng) —
// tránh nhầm với các cụm như "1. Chest findings:", "6. Unreported findings:"
// nằm giữa câu, vốn là 1 phần của nội dung chứ không phải tiêu đề section.
const VERIF_SECTION_RE = /^[ \t]*([A-Z][A-Z \/]{1,40}):[ \t]*\n+/gm

function splitVerificationSections(text: string): { title: string; body: string }[] {
  const parts = text.split(VERIF_SECTION_RE)
  if (parts.length === 1) return [{ title: '', body: text.trim() }]

  const sections: { title: string; body: string }[] = []
  if (parts[0].trim()) sections.push({ title: '', body: parts[0].trim() })
  for (let i = 1; i < parts.length; i += 2) {
    const body = (parts[i + 1] ?? '').trim()
    if (body) sections.push({ title: parts[i].toUpperCase(), body })
  }
  return sections
}

// Verification IMPRESSION thường là 1 đoạn liền chứa "1. ... 2. ... 3. ..." —
// tách thành từng dòng riêng nếu phát hiện dãy số thứ tự liên tục bắt đầu từ 1
const NUMBERED_ITEM_RE = /(?:^|\s)(\d+)\.\s+/g

function splitNumberedList(text: string): string[] | null {
  const matches = [...text.matchAll(NUMBERED_ITEM_RE)]
  if (matches.length < 2) return null
  for (let i = 0; i < matches.length; i++) {
    if (Number(matches[i][1]) !== i + 1) return null
  }

  const items: string[] = []
  for (let i = 0; i < matches.length; i++) {
    const start = (matches[i].index ?? 0) + matches[i][0].length
    const end = i + 1 < matches.length ? matches[i + 1].index : text.length
    items.push(text.slice(start, end).trim())
  }
  return items
}

// Verification thường chia theo chủ đề bằng các đoạn bắt đầu bằng
// "**Label**: ..." (vd "**Chest Findings:**", "**Nodule Characteristics**:")
// — gom các đoạn không có label vào group label gần nhất phía trước, để
// hiển thị thành từng khối riêng theo chủ đề. Nếu không có label nào,
// trả về 1 group duy nhất (label rỗng) để fallback render như cũ.
const LABEL_PARA_RE = /^\*\*(.+?)\*\*:?\s*/

function splitLabeledGroups(text: string): { label: string; paragraphs: string[] }[] {
  const paras = text.split(/\n{2,}/).map(p => p.trim()).filter(Boolean)
  const groups: { label: string; paragraphs: string[] }[] = []

  for (const para of paras) {
    const m = para.match(LABEL_PARA_RE)
    if (m) {
      groups.push({ label: m[1].replace(/:\s*$/, ''), paragraphs: [para.slice(m[0].length)] })
    } else if (groups.length > 0) {
      groups[groups.length - 1].paragraphs.push(para)
    } else {
      groups.push({ label: '', paragraphs: [para] })
    }
  }
  return groups
}

function Badge({ value, field }: { value: string | null; field: string }) {
  if (!value) return <span className="text-gray-600 text-xs">—</span>

  const isCritical =
    (field === 'lung_cancer_risk' && value !== 'No cancer within follow-up') ||
    (field === 'CVD_diagnosis' && value === 'Yes')

  const isAbnormal = [
    'chest_abn_54', 'chest_abn_55', 'chest_abn_56', 'chest_abn_57',
    'chest_abn_58', 'chest_abn_59', 'chest_abn_61', 'nodule_presence',
  ].includes(field) && value === 'Yes'

  const isNormal =
    value === 'No' ||
    value === 'No cancer within follow-up' ||
    value === 'Low risk'

  const cls = isCritical
    ? 'bg-red-500/15 text-red-400 border border-red-500/20'
    : isAbnormal
    ? 'bg-amber-500/15 text-amber-400 border border-amber-500/20'
    : isNormal
    ? 'bg-green-500/15 text-green-400 border border-green-500/20'
    : 'bg-blue-500/15 text-blue-400 border border-blue-500/20'

  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${cls}`}>
      {value}
    </span>
  )
}

function Row({ label, value, field }: { label: string; value: string | null; field: string }) {
  const question = TASK_QUESTION[field]
  return (
    <div className="py-1.5 border-b border-white/3 last:border-0">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm text-gray-400">{label}</span>
        <Badge value={value} field={field} />
      </div>
      {question && (
        <p className="text-[11px] text-sky-300/80 leading-snug mt-0.5 pr-2">{question}</p>
      )}
    </div>
  )
}

function SkeletonRow() {
  return (
    <div className="flex items-center justify-between py-2">
      <div className="h-3 w-32 skeleton" />
      <div className="h-5 w-16 skeleton rounded-full" />
    </div>
  )
}

interface CardProps {
  title: string
  icon: React.ReactNode
  accent?: string
  defaultOpen?: boolean
  loading?: boolean
  children: React.ReactNode
}

function AgentCard({ title, icon, accent, defaultOpen = false, loading = false, children }: CardProps) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className={`rounded-2xl border overflow-hidden transition-all duration-200 ${
      accent ?? 'border-white/8 bg-gray-900/60'
    }`}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/3 transition-colors"
      >
        <div className="flex items-center gap-2.5">
          <span className="text-blue-400">{icon}</span>
          <span className="font-medium text-sm text-white">{title}</span>
          {loading && (
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
          )}
        </div>
        {open
          ? <ChevronUp size={14} className="text-gray-600" />
          : <ChevronDown size={14} className="text-gray-600" />
        }
      </button>
      {open && <div className="px-4 pb-3">{children}</div>}
    </div>
  )
}

function EmptyCard({ title, icon, rowCount = 4 }: {
  title: string; icon: React.ReactNode; rowCount?: number
}) {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-2xl border border-white/5 bg-gray-900/40 overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/3 transition-colors"
      >
        <div className="flex items-center gap-2.5">
          <span className="text-gray-700">{icon}</span>
          <span className="font-medium text-sm text-gray-600">{title}</span>
        </div>
        {open
          ? <ChevronUp size={14} className="text-gray-700" />
          : <ChevronDown size={14} className="text-gray-700" />
        }
      </button>
      {open && (
        <div className="px-4 pb-4">
          <div className="flex flex-col items-center justify-center py-4 gap-2">
            <Lock size={18} className="text-gray-700" />
            <p className="text-xs text-gray-600 text-center">
              Add a patient to see results
            </p>
          </div>
          {Array.from({ length: rowCount }).map((_, i) => (
            <div key={i} className="flex items-center justify-between py-1.5 border-b border-white/3 last:border-0">
              <div className="h-2.5 w-28 bg-white/3 rounded" />
              <div className="h-4 w-12 bg-white/3 rounded-full" />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function LoadingCard({ title, icon, rowCount = 4 }: {
  title: string; icon: React.ReactNode; rowCount?: number
}) {
  return (
    <div className="rounded-2xl border border-white/8 bg-gray-900/60 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="text-blue-400/50">{icon}</span>
          <span className="font-medium text-sm text-gray-500">{title}</span>
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
        </div>
      </div>
      <div className="px-4 pb-3">
        {Array.from({ length: rowCount }).map((_, i) => (
          <SkeletonRow key={i} />
        ))}
      </div>
    </div>
  )
}

interface Props {
  result: AnalyzeResponse | null
  status: AnalysisStatus
  errorMsg: string | null
}

export default function AgentResults({ result, status, errorMsg }: Props) {

  if (status === 'idle' || status === 'error') {
    return (
      <main className="flex-1 overflow-y-auto space-y-2.5">
        {errorMsg && (
          <div className="mx-0.5 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-400 text-sm">
            {errorMsg}
          </div>
        )}
        <EmptyCard title="Radiology — Screening"     icon={<Stethoscope size={15} />} rowCount={8} />
        <EmptyCard title="Radiology — Nodule Detail" icon={<Stethoscope size={15} />} rowCount={4} />
        <EmptyCard title="Cardiology"                icon={<Heart size={15} />}       rowCount={2} />
        <EmptyCard title="Oncology"                  icon={<FlaskConical size={15} />} rowCount={1} />
        <EmptyCard title="Finding & Impression"      icon={<FileText size={15} />}    rowCount={2} />
        <EmptyCard title="Verification"              icon={<ShieldCheck size={15} />} rowCount={3} />
      </main>
    )
  }

  if (status === 'loading') {
    return (
      <main className="flex-1 overflow-y-auto space-y-2.5">
        <LoadingCard title="Radiology — Screening"     icon={<Stethoscope size={15} />} rowCount={8} />
        <LoadingCard title="Radiology — Nodule Detail" icon={<Stethoscope size={15} />} rowCount={4} />
        <LoadingCard title="Cardiology"                icon={<Heart size={15} />}       rowCount={2} />
        <LoadingCard title="Oncology"                  icon={<FlaskConical size={15} />} rowCount={1} />
        <LoadingCard title="Finding & Impression"      icon={<FileText size={15} />}    rowCount={2} />
        <LoadingCard title="Verification"              icon={<ShieldCheck size={15} />} rowCount={3} />
      </main>
    )
  }

  if (!result) return null

  const { radiology, cardiology, oncology, finding_impression, verification } = result

  return (
    <main className="flex-1 overflow-y-auto space-y-2.5">

      {/* Agent 1 — Radiology Screening */}
      <AgentCard title="Radiology — Screening" icon={<Stethoscope size={15} />} defaultOpen>
        {(Object.entries(radiology.screening.answer) as [string, string | null][]).map(([k, v]) => (
          <Row key={k} label={TASK_LABEL[k] ?? k} value={v} field={k} />
        ))}
      </AgentCard>

      {/* Agent 2 — Nodule Detail */}
      {radiology.detail && (
        <AgentCard
          title="Radiology — Nodule Detail"
          icon={<Stethoscope size={15} />}
          accent="border-amber-500/20 bg-amber-500/5"
          defaultOpen
        >
          {(Object.entries(radiology.detail.answer) as [string, string | null][]).map(([k, v]) => (
            <Row key={k} label={TASK_LABEL[k] ?? k} value={v} field={k} />
          ))}
        </AgentCard>
      )}

      {/* Agent 3 — Cardiology */}
      <AgentCard title="Cardiology" icon={<Heart size={15} />} defaultOpen>
        {(Object.entries(cardiology.answer) as [string, string | null][]).map(([k, v]) => (
          <Row key={k} label={TASK_LABEL[k] ?? k} value={v} field={k} />
        ))}
      </AgentCard>

      {/* Agent 4 — Oncology */}
      <AgentCard title="Oncology" icon={<FlaskConical size={15} />} defaultOpen>
        {(Object.entries(oncology.answer) as [string, string | null][]).map(([k, v]) => (
          <Row key={k} label={TASK_LABEL[k] ?? k} value={v} field={k} />
        ))}
      </AgentCard>

      {/* Agent 4b — Finding & Impression */}
      <AgentCard title="Finding & Impression" icon={<FileText size={15} />} defaultOpen>
        {finding_impression.findings && (
          <div className="mb-3">
            <p className="text-xs font-semibold text-sky-400 uppercase tracking-wider mb-1.5 border-l-2 border-sky-500/40 pl-2">Findings</p>
            <p className="text-sm text-gray-300 leading-relaxed">{renderAnnotated(finding_impression.findings)}</p>
          </div>
        )}
        {finding_impression.impression && (
          <div>
            <p className="text-xs font-semibold text-sky-400 uppercase tracking-wider mb-1.5 border-l-2 border-sky-500/40 pl-2">Impression</p>
            <p className="text-sm text-gray-300 leading-relaxed">{renderAnnotated(finding_impression.impression)}</p>
          </div>
        )}
      </AgentCard>

      {/* Agent 5 — Verification */}
      <AgentCard
        title="Verification — Independent Review"
        icon={<ShieldCheck size={15} />}
        accent="border-blue-500/20 bg-blue-500/5"
        defaultOpen
      >
        {verification.analysis ? (
          splitVerificationSections(verification.analysis).map((section, i) => (
            <div key={i} className={i > 0 ? 'mt-3' : undefined}>
              {section.title && (
                <p className="text-xs font-semibold text-sky-400 uppercase tracking-wider mb-1.5 border-l-2 border-sky-500/40 pl-2">
                  {section.title}
                </p>
              )}
              {(() => {
                const groups = splitLabeledGroups(section.body)
                if (groups.length > 1 || groups[0]?.label) {
                  return (
                    <div className="space-y-3">
                      {groups.map((g, gi) => (
                        <div key={gi}>
                          {g.label && (
                            <p className="text-base font-bold text-pink-400 mb-1.5">
                              {g.label}
                            </p>
                          )}
                          {g.paragraphs.map((p, pi) => (
                            <p key={pi} className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap mb-1.5 last:mb-0">
                              {renderAnnotated(p)}
                            </p>
                          ))}
                        </div>
                      ))}
                    </div>
                  )
                }

                const items = splitNumberedList(section.body)
                if (!items) {
                  return (
                    <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                      {renderAnnotated(section.body)}
                    </p>
                  )
                }
                return (
                  <ol className="space-y-1.5">
                    {items.map((item, j) => (
                      <li key={j} className="text-sm text-gray-300 leading-relaxed flex gap-2">
                        <span className="text-gray-600">{j + 1}.</span>
                        <span>{renderAnnotated(item)}</span>
                      </li>
                    ))}
                  </ol>
                )
              })()}
            </div>
          ))
        ) : (
          <p className="text-sm text-gray-600 italic">No analysis available.</p>
        )}
      </AgentCard>

    </main>
  )
}
