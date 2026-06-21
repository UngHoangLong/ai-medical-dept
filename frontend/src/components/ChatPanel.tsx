import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, MessageSquare, Loader2, Mic } from 'lucide-react'
import type { AnalyzeResponse, ChatMessage } from '../types/api'
import type { AnalysisStatus } from '../App'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { transcribeAudio } from '../lib/patientStore'

// DEV: Dùng "" → request đi qua Vite proxy (Node.js, same-origin, không bị browser buffer)
// PROD: Dùng URL thật từ env hoặc Caddy reverse proxy cũng route same-origin
const CHAT_BACKEND = import.meta.env.VITE_CHAT_SERVICE_URL ?? ""


interface Props {
  result: AnalyzeResponse | null
  reportId: string | null
  cacheKey: string
  status: AnalysisStatus
  className?: string
  style?: React.CSSProperties
}

export default function ChatPanel({ result, reportId, cacheKey, status, className, style }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [agentStatus, setAgentStatus] = useState<string>('')
  const [isStreaming, setIsStreaming] = useState(false) // true = đang nhận token → hiển thị text thường (nhanh)
  const bottomRef = useRef<HTMLDivElement>(null)

  const activeReportId = reportId ?? 'unknown_report_id'

  // Speech-to-Text State & Refs
  const [isRecording, setIsRecording] = useState(false)
  const [sttLoading, setSttLoading] = useState(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  // Clean up recording resources on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  async function toggleRecording() {
    if (isRecording) {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop()
      }
      setIsRecording(false)
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        streamRef.current = stream
        audioChunksRef.current = []

        const mediaRecorder = new MediaRecorder(stream)
        mediaRecorderRef.current = mediaRecorder

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data)
          }
        }

        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' })

          // Release microphone track
          if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop())
            streamRef.current = null
          }

          setSttLoading(true)
          try {
            const responseText = await transcribeAudio(audioBlob)
            if (responseText) {
              setInput(prev => (prev ? `${prev} ${responseText}` : responseText))
            }
          } catch (err) {
            console.error('Failed to transcribe audio:', err)
          } finally {
            setSttLoading(false)
          }
        }

        mediaRecorder.start()
        setIsRecording(true)
      } catch (err) {
        console.error('Failed to access microphone:', err)
        alert('Could not access microphone. Please check permissions.')
      }
    }
  }

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

  // 2. GỬI TIN NHẮN VỚI STREAMING (SSE qua XMLHttpRequest)
  // ====================================================================
  // Dùng XHR thay cho fetch+ReadableStream.
  // XHR.onprogress fire incremental khi có data đến, không bị ảnh hưởng bởi
  // HTTP/2 connection pooling hay idle-connection buffering của browser như fetch.
  // ====================================================================
  const tokenQueueRef = useRef<string[]>([])
  const visibleTextRef = useRef('')
  const rafIdRef = useRef<number>(0)
  const streamDoneRef = useRef(false)
  const xhrRef = useRef<XMLHttpRequest | null>(null)

  function startTypewriter() {
    function tick() {
      if (tokenQueueRef.current.length > 0) {
        // Gom tất cả token hiện có vào hiển thị ngay lập tức (không delay từng chữ)
        visibleTextRef.current += tokenQueueRef.current.join('')
        tokenQueueRef.current = []

        const text = visibleTextRef.current
        setMessages(prev => {
          const newMsgs = [...prev]
          newMsgs[newMsgs.length - 1] = { ...newMsgs[newMsgs.length - 1], content: text }
          return newMsgs
        })
      }

      if (tokenQueueRef.current.length > 0 || !streamDoneRef.current) {
        rafIdRef.current = requestAnimationFrame(tick)
      }
    }

    rafIdRef.current = requestAnimationFrame(tick)
  }

  function sendMessage() {
    const text = input.trim()
    if (!text || loading || status !== 'done') return

    // Huỷ request cũ nếu đang chạy
    if (xhrRef.current) {
      xhrRef.current.abort()
      xhrRef.current = null
    }

    setMessages(prev => [
      ...prev,
      { role: 'user', content: text },
      { role: 'assistant', content: '' }
    ])
    setInput('')
    setLoading(true)
    setIsStreaming(true)
    setAgentStatus('Đang khởi tạo...')

    tokenQueueRef.current = []
    visibleTextRef.current = ''
    streamDoneRef.current = false
    startTypewriter()

    // Dùng XHR — onprogress fire incremental theo từng chunk data đến,
    // không bị browser buffer như fetch trên HTTP/2 idle connection.
    const xhr = new XMLHttpRequest()
    xhrRef.current = xhr
    let processedLength = 0
    let sseBuffer = ''

    function processSSEChunk(newText: string) {
      sseBuffer += newText
      const events = sseBuffer.split(/\n\n/)
      sseBuffer = events.pop() ?? ''

      for (const event of events) {
        for (const line of event.split('\n')) {
          if (!line.startsWith('data:')) continue
          const dataStr = line.slice(5).trim()
          if (!dataStr || dataStr === '[DONE]') continue
          try {
            const payload = JSON.parse(dataStr)
            if (payload.type === 'token') {
              tokenQueueRef.current.push(payload.content)
              setAgentStatus('')
            } else if (payload.type === 'status') {
              setAgentStatus(payload.message)
            } else if (payload.type === 'error') {
              console.error('LangGraph error:', payload.message)
              setAgentStatus('Có lỗi xảy ra!')
            } else if (payload.type === 'done') {
              setAgentStatus('')
            }
          } catch {
            // JSON chưa đủ — bỏ qua
          }
        }
      }
    }

    function finish() {
      streamDoneRef.current = true
      xhrRef.current = null
      const waitForQueue = () => {
        if (tokenQueueRef.current.length === 0) {
          setLoading(false)
          setIsStreaming(false)
          setAgentStatus('')
        } else {
          requestAnimationFrame(waitForQueue)
        }
      }
      requestAnimationFrame(waitForQueue)
    }

    xhr.open('POST', `${CHAT_BACKEND}/api/v1/chat`, true)
    xhr.setRequestHeader('Content-Type', 'application/json')
    xhr.setRequestHeader('Accept', 'text/event-stream')

    // onprogress: fire mỗi khi có thêm data — đây là điểm khác biệt so với fetch
    xhr.onprogress = () => {
      const newChunk = xhr.responseText.slice(processedLength)
      processedLength = xhr.responseText.length
      if (newChunk) processSSEChunk(newChunk)
    }

    xhr.onload = () => {
      // Flush phần còn lại sau khi response kết thúc
      const remaining = xhr.responseText.slice(processedLength)
      if (remaining) processSSEChunk(remaining)
      finish()
    }

    xhr.onerror = () => {
      tokenQueueRef.current = ['⚠️ Lỗi kết nối tới Server Chat.']
      finish()
    }

    xhr.onabort = () => {
      finish()
    }

    xhr.send(JSON.stringify({
      query: text,
      report_id: activeReportId,
      user_id: 'doctor_01',
    }))
  }


  const suggestions = result ? [
    ...(result.radiology?.detail ? ['Follow-up for this nodule?'] : []),
    ...(result.cardiology?.answer?.CVD_diagnosis === 'Yes' ? ['Why CVD positive?'] : []),
    'Summarize key findings',
    'What does Agent 5 say?',
  ].slice(0, 3) : []

  const isDisabled = status !== 'done'

  return (
    <aside
      style={style}
      className={`flex flex-col bg-gray-900/60 rounded-2xl border border-white/8 overflow-hidden ${className || ''}`}
    >
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
                <div className={`shrink-0 w-6 h-6 rounded-xl flex items-center justify-center ${msg.role === 'assistant' ? 'bg-blue-500/20' : 'bg-white/10'
                  }`}>
                  {msg.role === 'assistant'
                    ? <Bot size={11} className="text-blue-400" />
                    : <User size={11} className="text-gray-400" />
                  }
                </div>

                {/* Đã sửa: Chỉ giữ lại 1 thẻ div bọc nội dung */}
                <div className={`max-w-[85%] px-3 py-2 rounded-2xl text-xs leading-relaxed ${msg.role === 'assistant'
                  ? 'bg-white/5 text-gray-200 rounded-tl-sm'
                  : 'bg-blue-500/20 text-blue-100 rounded-tr-sm'
                  } whitespace-pre-wrap overflow-hidden`}>

                  {msg.role === 'user' ? (
                    msg.content
                  ) : (
                    <div className="markdown-body text-xs prose prose-invert max-w-none prose-p:leading-relaxed prose-pre:bg-gray-800 prose-th:border-gray-600 prose-td:border-gray-700">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content + (isStreaming && i === messages.length - 1 ? ' ▍' : '')}
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
          <button
            onClick={toggleRecording}
            disabled={isDisabled || sttLoading}
            title={isRecording ? 'Stop recording' : 'Record voice note'}
            type="button"
            className={`w-8 h-8 rounded-xl flex items-center justify-center transition-all shrink-0 ${isRecording
              ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse'
              : sttLoading
                ? 'bg-gray-800 text-blue-400'
                : 'bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-white border border-white/10'
              }`}
          >
            {sttLoading ? (
              <Loader2 size={13} className="animate-spin" />
            ) : (
              <Mic size={13} />
            )}
          </button>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
            }}
            disabled={isDisabled || loading || isRecording}
            placeholder={isDisabled ? 'Run analysis first...' : isRecording ? 'Recording audio...' : 'Ask about this scan...'}
            rows={1}
            className="flex-1 text-xs bg-gray-800 border border-white/10 rounded-xl px-3 py-2 resize-none text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500/40 disabled:opacity-30 transition-all"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || loading || isDisabled || isRecording || sttLoading}
            className="w-8 h-8 rounded-xl bg-blue-500 hover:bg-blue-400 disabled:bg-white/5 disabled:text-gray-600 text-white flex items-center justify-center transition-all shrink-0"
          >
            <Send size={13} />
          </button>
        </div>
      </div>
    </aside>
  )
}