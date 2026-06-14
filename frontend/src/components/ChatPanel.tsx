import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, MessageSquare, Loader2 } from 'lucide-react'
import type { AnalyzeResponse, ChatMessage } from '../types/api'
import type { AnalysisStatus } from '../App'

const BACKEND = import.meta.env.VITE_BACKEND_URL ?? ''

interface Props {
  result: AnalyzeResponse | null
  reportId: string | null
  cacheKey: string
  status: AnalysisStatus
}

export default function ChatPanel({ result, reportId, cacheKey, status }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Reset when a new analysis starts or the active patient changes
  useEffect(() => {
    if (status === 'loading') {
      setMessages([])
    } else if (status === 'done') {
      setMessages([{
        role: 'assistant',
        content: 'Analysis complete! Ask me anything about this patient\'s CT scan results.',
      }])
    } else {
      setMessages([])
    }
  }, [status, cacheKey]) // eslint-disable-line react-hooks/exhaustive-deps

  async function sendMessage() {
    const text = input.trim()
    if (!text || loading || status !== 'done') return

    setMessages(prev => [...prev, { role: 'user', content: text }])
    setInput('')
    setLoading(true)

    try {
      const resp = await fetch(`${BACKEND}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: text,
          report_id: reportId ?? result?.report_id ?? 'unknown_report_id',
          user_id: null,
        }),
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }])
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '⚠️ Chat endpoint not yet available.',
      }])
    } finally {
      setLoading(false)
    }
  }

  const suggestions = result ? [
    ...(result.radiology.detail ? ['Follow-up for this nodule?'] : []),
    ...(result.cardiology.answer.CVD_diagnosis === 'Yes' ? ['Why CVD positive?'] : []),
    'Summarize key findings',
    'What does Agent 5 say?',
  ].slice(0, 3) : []

  // ── Empty / loading states ─────────────────────────────────────
  const isDisabled = status !== 'done'

  return (
    <aside className="w-72 shrink-0 flex flex-col bg-gray-900/60 rounded-2xl border border-white/8 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/5 flex items-center gap-2">
        <MessageSquare size={14} className="text-gray-500" />
        <span className="text-xs font-medium text-gray-400">Ask AI</span>
        {status === 'done' && (
          <span className="ml-auto w-1.5 h-1.5 rounded-full bg-green-400" />
        )}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-3">

        {status === 'idle' && (
          <div className="h-full flex flex-col items-center justify-center text-center px-4 gap-3">
            <div className="w-12 h-12 rounded-2xl bg-white/3 border border-white/8 flex items-center justify-center">
              <Bot size={20} className="text-gray-700" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">No active session</p>
              <p className="text-xs text-gray-700 mt-1 leading-relaxed">
                Add a patient and run<br />analysis to start chatting
              </p>
            </div>
          </div>
        )}

        {status === 'loading' && (
          <div className="h-full flex flex-col items-center justify-center text-center px-4 gap-3">
            <div className="w-12 h-12 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
              <Loader2 size={20} className="text-blue-400 animate-spin" />
            </div>
            <p className="text-xs text-gray-500">Waiting for analysis...</p>
          </div>
        )}

        {status === 'error' && (
          <div className="h-full flex flex-col items-center justify-center text-center px-4 gap-3">
            <div className="w-12 h-12 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center">
              <Bot size={20} className="text-red-600" />
            </div>
            <p className="text-xs text-gray-600">Analysis failed. Try again.</p>
          </div>
        )}

        {status === 'done' && (
          <div className="space-y-3">
            {messages.map((msg, i) => (
              <div key={i} className={`flex gap-2 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`shrink-0 w-6 h-6 rounded-xl flex items-center justify-center ${
                  msg.role === 'assistant' ? 'bg-blue-500/20' : 'bg-white/10'
                }`}>
                  {msg.role === 'assistant'
                    ? <Bot size={11} className="text-blue-400" />
                    : <User size={11} className="text-gray-400" />
                  }
                </div>
                <div className={`max-w-[85%] px-3 py-2 rounded-2xl text-xs leading-relaxed ${
                  msg.role === 'assistant'
                    ? 'bg-white/5 text-gray-200 rounded-tl-sm'
                    : 'bg-blue-500/20 text-blue-100 rounded-tr-sm'
                }`}>
                  {msg.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex gap-2">
                <div className="w-6 h-6 rounded-xl bg-blue-500/20 flex items-center justify-center shrink-0">
                  <Bot size={11} className="text-blue-400" />
                </div>
                <div className="bg-white/5 px-3 py-2.5 rounded-2xl rounded-tl-sm flex gap-1 items-center">
                  {[0, 150, 300].map(d => (
                    <span
                      key={d}
                      className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce"
                      style={{ animationDelay: `${d}ms` }}
                    />
                  ))}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Suggestions */}
      {status === 'done' && suggestions.length > 0 && messages.length <= 1 && (
        <div className="px-3 pb-2 flex flex-wrap gap-1.5">
          {suggestions.map((s, i) => (
            <button
              key={i}
              onClick={() => setInput(s)}
              className="text-xs bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 border border-blue-500/20 px-2.5 py-1 rounded-full transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="p-3 border-t border-white/5">
        <div className="flex gap-2 items-end">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
            }}
            disabled={isDisabled}
            placeholder={isDisabled ? 'Run analysis first...' : 'Ask about this scan...'}
            rows={1}
            className="flex-1 text-xs bg-gray-800 border border-white/10 rounded-xl px-3 py-2 resize-none text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500/40 disabled:opacity-30 transition-all"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || loading || isDisabled}
            className="w-8 h-8 rounded-xl bg-blue-500 hover:bg-blue-400 disabled:bg-white/5 disabled:text-gray-600 text-white flex items-center justify-center transition-all shrink-0"
          >
            <Send size={13} />
          </button>
        </div>
      </div>
    </aside>
  )
}
