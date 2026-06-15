import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, MessageSquare, Loader2 } from 'lucide-react'
import type { AnalyzeResponse, ChatMessage } from '../types/api'
import type { AnalysisStatus } from '../App'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const CHAT_BACKEND = import.meta.env.VITE_CHAT_SERVICE_URL || 'http://localhost:8000'

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
  const [agentStatus, setAgentStatus] = useState<string>('') // Thêm state để hiển thị tiến trình (status node)
  const bottomRef = useRef<HTMLDivElement>(null)

  const activeReportId = reportId ?? 'unknown_report_id'

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, agentStatus])

  // 1. CẬP NHẬT LOGIC TẢI LỊCH SỬ CHAT
// 1. CẬP NHẬT LOGIC TẢI LỊCH SỬ CHAT (Đã fix Race Condition)
  useEffect(() => {
    // Cờ báo hiệu component còn sống, chống React gọi 2 lần đè state lẫn nhau
    let isMounted = true; 

    async function fetchHistory() {
      if (status === 'done' && activeReportId !== 'unknown_report_id') {
        try {
          console.log(`Đang gọi API lấy lịch sử cho report: ${activeReportId}...`);
          
          const res = await fetch(`${CHAT_BACKEND}/api/v1/history/${activeReportId}`)
          if (res.ok) {
            const data = await res.json()
            console.log("Dữ liệu API trả về:", data); // Check xem log có ra không
            
            // Chỉ update state nếu component chưa bị unmount
            if (isMounted) {
              if (data.messages && data.messages.length > 0) {
                console.log("-> Đã set lịch sử chat!");
                setMessages(data.messages)
              } else {
                console.log("-> Không có lịch sử, set câu chào mặc định.");
                setMessages([{
                  role: 'assistant',
                  content: 'Analysis complete! Ask me anything about this patient\'s CT scan results.',
                }])
              }
            }
            return; // Thoát hàm thành công
          }
        } catch (error) {
          console.error('Lỗi khi tải lịch sử chat:', error)
        }
        
        // Nhánh fallback: Nếu API sập hoặc lỗi, vẫn set câu chào mặc định (nếu còn mounted)
        if (isMounted) {
          setMessages([{
            role: 'assistant',
            content: 'Analysis complete! Ask me anything about this patient\'s CT scan results.',
          }])
        }

      } else if (status === 'loading') {
        if (isMounted) setMessages([])
      }
    }

    fetchHistory()

    // Cleanup function: Khi dependency thay đổi hoặc component hủy, bật cờ false
    return () => {
      isMounted = false;
    }
  }, [status, activeReportId, cacheKey])

  // 2. CẬP NHẬT LOGIC GỬI TIN NHẮN VỚI STREAMING (SSE)
  async function sendMessage() {
    const text = input.trim()
    if (!text || loading || status !== 'done') return

    // Thêm tin nhắn của user và chuẩn bị sẵn một tin nhắn rỗng cho assistant
    setMessages(prev => [
      ...prev, 
      { role: 'user', content: text },
      { role: 'assistant', content: '' } // Khởi tạo cục message rỗng để đắp text dần vào
    ])
    setInput('')
    setLoading(true)
    setAgentStatus('Đang khởi tạo...')

    try {
      const resp = await fetch(`${CHAT_BACKEND}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: text,
          report_id: activeReportId,
          user_id: 'doctor_01', // Thay bằng ID của user đang login nếu có
        }),
      })

      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`)

      // Đọc luồng dữ liệu trả về
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let assistantMessage = ''
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        // Tách các chunk dữ liệu cách nhau bởi 2 dấu xuống dòng (chuẩn SSE)
        const parts = buffer.split('\n\n')
        buffer = parts.pop() || '' // Giữ lại phần chưa hoàn thiện ở cuối buffer

        for (const part of parts) {
          if (part.startsWith('data: ')) {
            const dataStr = part.slice(6) // Cắt bỏ chữ "data: "

            try {
              const payload = JSON.parse(dataStr)

              if (payload.type === 'token') {
                // Khi nhận được text từ LLM
                setAgentStatus('') // Ẩn status đi vì đã bắt đầu gõ text
                assistantMessage += payload.content
                setMessages(prev => {
                  const newMsgs = [...prev]
                  newMsgs[newMsgs.length - 1].content = assistantMessage
                  return newMsgs
                })
              } else if (payload.type === 'status') {
                // Cập nhật trạng thái Node đang chạy (vd: Đang tra cứu FDA...)
                setAgentStatus(payload.message)
              } else if (payload.type === 'error') {
                console.error("Lỗi LangGraph:", payload.message)
                setAgentStatus('Có lỗi xảy ra!')
              } else if (payload.type === 'done') {
                // Xử lý xong
                setAgentStatus('')
              }
            } catch (err) {
              // Bỏ qua lỗi JSON parse cho các gói tin có thể bị rách giữa chừng
            }
          }
        }
      }
    } catch (err) {
      setMessages(prev => {
        const newMsgs = [...prev]
        newMsgs[newMsgs.length - 1].content = '⚠️ Lỗi kết nối tới Server Chat.'
        return newMsgs
      })
    } finally {
      setLoading(false)
      setAgentStatus('')
    }
  }

  const suggestions = result ? [
    ...(result.radiology?.detail ? ['Follow-up for this nodule?'] : []),
    ...(result.cardiology?.answer?.CVD_diagnosis === 'Yes' ? ['Why CVD positive?'] : []),
    'Summarize key findings',
    'What does Agent 5 say?',
  ].slice(0, 3) : []

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
                
                {/* Đã sửa: Chỉ giữ lại 1 thẻ div bọc nội dung */}
                <div className={`max-w-[85%] px-3 py-2 rounded-2xl text-xs leading-relaxed ${
                  msg.role === 'assistant'
                    ? 'bg-white/5 text-gray-200 rounded-tl-sm'
                    : 'bg-blue-500/20 text-blue-100 rounded-tr-sm'
                } whitespace-pre-wrap overflow-hidden`}>
                  
                  {/* Nếu là User thì in text thường, nếu là Bot thì Parse Markdown */}
                  {msg.role === 'user' ? (
                    msg.content
                  ) : (
                    /* CHUYỂN CLASSNAME LÊN THẺ DIV BỌC NGOÀI */
                    <div className="markdown-body text-xs prose prose-invert max-w-none prose-p:leading-relaxed prose-pre:bg-gray-800 prose-th:border-gray-600 prose-td:border-gray-700">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  )}

                </div>
              </div>
            ))}

            {/* Hiển thị thanh tiến trình của các Agent dưới nền (Trạng thái Node) */}
            {loading && agentStatus && (
               <div className="flex gap-2">
                 <div className="w-6 h-6 rounded-xl bg-transparent flex items-center justify-center shrink-0" />
                 <span className="text-[10px] text-blue-400 italic flex items-center gap-1.5">
                   <Loader2 size={10} className="animate-spin" /> {agentStatus}
                 </span>
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
            disabled={isDisabled || loading}
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