# 1. Sử dụng image Python phiên bản gọn nhẹ
FROM python:3.11-slim

# 2. Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# 3. Copy file requirements vào trước để cài đặt thư viện
COPY requirements.txt .

# Cài đặt thư viện (thêm --no-cache-dir để image nhẹ hơn)
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy toàn bộ code hiện tại (bao gồm main.py và thư mục custom_tools) vào container
COPY . .

# 5. Mở cổng 8000 để giao tiếp ra bên ngoài
EXPOSE 8000

# 6. Lệnh để khởi chạy server khi container bật lên
CMD ["python", "server.py"]