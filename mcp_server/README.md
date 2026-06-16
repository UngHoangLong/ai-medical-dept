# MCP Server (Model Context Protocol)

Dịch vụ MCP Server được xây dựng dựa trên nền tảng **FastMCP** (Python SDK), cung cấp các công cụ (tools) truy vấn trực tiếp cơ sở dữ liệu của Cục quản lý Thực phẩm và Dược phẩm Hoa Kỳ (OpenFDA) để hỗ trợ các LLM Agent kiểm tra tương tác thuốc và hồ sơ an toàn y tế.

## 🚀 Tính năng chính
- **Chuẩn MCP (Model Context Protocol)**: Triển khai các công cụ theo đặc tả MCP giúp dễ dàng tích hợp với bất kỳ LLM Agent nào (như LangGraph, Claude Desktop, Cursor...).
- **Tìm kiếm Song song (Multi-threading)**: Sử dụng `ThreadPoolExecutor` để gửi song song tối đa 5 truy vấn đồng thời lên API OpenFDA, tăng tốc độ xử lý khi phân tích danh sách nhiều loại thuốc.
- **Tích hợp OpenFDA**: Tìm kiếm thông tin thuốc theo cả tên thương mại (brand name) và tên gốc/tên hoạt chất (generic name).

## 🔌 Danh sách Công cụ (Tools)

### 1. `get_drug_interactions`
- **Mô tả**: Tải và tổng hợp thông tin tương tác thuốc, cảnh báo đặc biệt (boxed warning), chống chỉ định (contraindications) từ nhãn FDA của danh sách thuốc được cung cấp.
- **Tham số đầu vào**: `drugs` (danh sách chuỗi tên thuốc, ví dụ: `["ibuprofen", "citalopram"]`).
- **Đầu ra**: Chuỗi văn bản chứa thông tin chi tiết các mục cảnh báo tương tác của từng thuốc phục vụ cho Agent đối chiếu chéo.

### 2. `get_drug_safety_profile`
- **Mô tả**: Trích xuất hồ sơ an toàn chi tiết bao gồm chỉ định, liều lượng, phản ứng phụ, lưu ý cho các nhóm đối tượng đặc biệt (phụ nữ mang thai, người cao tuổi, trẻ em).
- **Tham số đầu vào**: `drugs` (danh sách chuỗi tên thuốc).
- **Đầu ra**: Báo cáo tổng quan về độ an toàn của thuốc dưới dạng cấu trúc JSON hoặc văn bản định dạng sẵn.

## 🛠️ Cài đặt & Chạy ứng dụng

### 1. Chạy cục bộ (Local Development)

**Cài đặt thư viện:**
```bash
pip install -r requirements.txt
```

**Khởi chạy server:**
```bash
python server.py
```
Mặc định, server sẽ chạy HTTP Transport tại cổng `8000`: `http://localhost:8000/mcp`

### 2. Chạy bằng Docker
Bạn có thể build và khởi chạy ứng dụng độc lập bằng Docker:

```bash
# Build image
docker build -t ai-medical-mcp-server:latest .

# Chạy container trên cổng 8002
docker run -d -p 8002:8000 ai-medical-mcp-server:latest
```

## 📁 Cấu trúc thư mục
```text
mcp_server/
├── custom_tools/
│   ├── fda_requests.py       # Module thực thi các cuộc gọi API OpenFDA song song
│   └── fda_multi_results.txt # Bản ghi kết quả chạy thử mẫu
├── Dockerfile                # Dockerfile tối giản hóa dung lượng
├── server.py                 # File chạy chính của FastMCP
├── requirements.txt          # Khai báo thư viện (fastmcp, requests)
└── README.md                 # Tài liệu này
```
