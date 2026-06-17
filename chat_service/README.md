# Chat Service

Dịch vụ Chat Agent được xây dựng dựa trên **FastAPI** và **LangGraph**, chịu trách nhiệm điều phối luồng trò chuyện y khoa, truy vấn thông tin bổ trợ từ MCP Server (FDA) và lưu trữ lịch sử hội thoại vào cơ sở dữ liệu PostgreSQL.

## 🚀 Tính năng chính
- **LangGraph Agent**: Sử dụng LangGraph để xây dựng đồ thị tác nhân thông minh, hỗ trợ gọi các công cụ y khoa động (qua MCP Server).
- **DeepSeek Integration**: Tích hợp mô hình DeepSeek qua endpoint API của DeepSeek (`https://api.deepseek.com`).
- **Postgres Checkpointer**: Lưu trữ và khôi phục trạng thái hội thoại (lịch sử chat, biến trạng thái) theo từng `thread_id` (trùng với `report_id`) thông qua `AsyncPostgresSaver` của LangGraph.
- **Server Sent Events (SSE)**: Stream dữ liệu câu trả lời thời gian thực từ Agent đến Client.

## 🛠️ Cài đặt & Chạy ứng dụng

### 1. Chuẩn bị môi trường
Tạo file `.env` tại thư mục này từ file mẫu:
```env
DEEPSEEK_API_KEY="sk-..."
PGHOST=127.0.0.1
PGUSER=postgres
PGPORT=5432
PGDATABASE=rimine
PGPASSWORD=postgres
DB_URI=postgresql://<host>:<port>/<dbname>?user=<user>&password=<password>
MCP_SERVER_URL=http://localhost:8002/mcp
ENABLE_TRACING=False
```

### 2. Chạy cục bộ (Local Development)

**Cài đặt thư viện:**
```bash
pip install -r requirements.txt
```

**Khởi chạy server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```
Dịch vụ sẽ khả dụng tại: `http://localhost:8001`

### 3. Chạy bằng Docker
Bạn có thể build và chạy thông qua Dockerfile đi kèm:

```bash
# Build image
docker build -t ai-medical-chat-service:latest .

# Run container
docker run -d -p 8001:8000 --env-file .env ai-medical-chat-service:latest
```

## 📁 Cấu trúc thư mục
```text
chat_service/
├── app/
│   ├── api/v1/        # Định nghĩa các endpoint FastAPI (Chat, History, Threads)
│   ├── config/        # Quản lý cấu hình thông qua Pydantic Settings
│   ├── graphs/        # Cấu trúc đồ thị LangGraph (nodes, edges, state)
│   ├── prompts/       # Định nghĩa System Prompt và các mẫu prompt
│   ├── utils/         # Hàm bổ trợ (truy vấn ngữ cảnh y tế, kết nối DB)
│   └── main.py        # Điểm khởi chạy FastAPI, thiết lập DB Pool & MCP Client
├── Dockerfile         # Dockerfile tối ưu hóa dung lượng (python-slim)
├── requirements.txt   # Danh sách thư viện phụ thuộc
└── README.md          # Tài liệu này
```

## 🔌 API Endpoints

### 1. Gửi tin nhắn và Stream câu trả lời (SSE)
- **Endpoint**: `POST /api/v1/chat`
- **Payload**:
  ```json
  {
    "query": "Thuốc ibuprofen có tương tác gì với citalopram không?",
    "report_id": "uuid_của_báo_cáo",
    "user_id": "doctor_01"
  }
  ```
- **Response**: Trả về luồng dữ liệu `text/event-stream` chứa các chunk dạng `data: {"type": "token", "content": "..."}` hoặc cập nhật trạng thái xử lý `data: {"type": "status", "message": "..."}`.

### 2. Lấy lịch sử hội thoại
- **Endpoint**: `GET /api/v1/history/{thread_id}`
- **Response**:
  ```json
  {
    "messages": [
      { "role": "user", "content": "..." },
      { "role": "assistant", "content": "..." }
    ]
  }
  ```

### 3. Lấy danh sách các phiên chat
- **Endpoint**: `GET /api/v1/threads`
- **Response**: Trả về danh sách các `thread_id` đang có trong hệ thống, sắp xếp theo checkpoint mới nhất.
