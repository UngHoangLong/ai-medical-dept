# Sử dụng phiên bản Python slim để giảm dung lượng image
FROM python:3.11-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Đặt biến môi trường tối ưu cho Python trong Docker
# Không ghi các file .pyc ra đĩa
ENV PYTHONDONTWRITEBYTECODE=1 \
# Bắn log thẳng ra terminal (không bị buffer delay)
    PYTHONUNBUFFERED=1

# Cài đặt các gói hệ thống cần thiết (Cực kỳ quan trọng cho thư viện psycopg kết nối Postgres)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy file danh sách thư viện và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn dự án vào container
COPY . .

# Mở cổng 8000 cho FastAPI
EXPOSE 8000

# Lệnh khởi chạy server khi container start
# (Giả định file chứa biến 'app = FastAPI(...)' nằm ở app/main.py)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]