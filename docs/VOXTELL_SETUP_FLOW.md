# VoxTell Modal Setup & Run Flow

## 1. Luồng chạy tổng quát

### Luồng trên Modal

```txt
User mở Modal URL
→ Nginx port 8000
→ Frontend Vite port 2811
→ Backend FastAPI port 1711
→ server.py xử lý API
→ VoxTell load model
→ S3 lấy DICOM zip
→ convert DICOM → NIfTI
→ NiiVue hiển thị CT volume
→ user nhập prompt
→ backend chạy segmentation
→ trả mask .nii.gz về frontend
```

### Luồng CT Viewer

```txt
App.tsx
→ truyền pid + series_uid vào CTViewer
→ CTViewer tự gọi backend để tải CT volume
→ Viewer dùng NiiVue load .nii.gz
→ user chọn Axial / Coronal / Sagittal / Multi
→ user nhập prompt
→ Run Segmentation
→ backend trả segmentation mask
→ CTViewer thêm mask vào danh sách segmentations[]
→ có thể hide/show/download/remove từng mask
```

---

## 2. Thứ tự setup khi đổi sang Modal account mới

### Bước 1: Login Modal account mới

```bash
modal token new
```

Kiểm tra account hiện tại:

```bash
modal profile current
```

---

### Bước 2: Tạo lại Modal Secret từ `.env`

Modal Secret nằm theo từng account/workspace, nên đổi account là phải tạo lại.

```bash
PYTHONPATH="$(pwd)/.certfix" modal secret create aws-s3 --from-dotenv .env --force
```

`.env` tối thiểu cần có:

```env
S3_BUCKET_NAME=...
AWS_REGION=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Nếu `server.py` cần DB/Postgres thì `.env` cũng phải có thêm biến DB tương ứng, ví dụ:

```env
DATABASE_URL=...
POSTGRES_URL=...
```

Tùy code hiện tại đang đọc tên biến nào thì giữ đúng tên đó.

Kiểm tra secret:

```bash
PYTHONPATH="$(pwd)/.certfix" modal secret list
```

---

### Bước 3: Bảo đảm image Modal có đủ dependency

Vì Modal account mới build image lại từ đầu, cần cài đủ package mà `server.py` import.

Trong `modal_web_demo_fine.py`, phần `.run_commands(...)`, cần có:

```python
"bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install boto3 python-dotenv'",
"bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install asyncpg'",
```

Lỗi đã gặp:

```txt
ModuleNotFoundError: No module named 'asyncpg'
```

Cách sửa là thêm `asyncpg`, không phải đổi URL.

---

### Bước 4: Giữ Modal chạy `server.py`

Hướng đang dùng hiện tại:

```txt
Modal backend vẫn chạy serving.backend.server:app
```

Lệnh trong `run.sh` hoặc script tương đương phải là:

```bash
conda run -n voxtell python -m uvicorn serving.backend.server:app --host 0.0.0.0 --port 1711
```

Không đổi sang file khác nếu muốn tiếp tục dùng `server.py`.

---

### Bước 5: Commit và push nếu Modal clone từ GitHub

Trong `modal_web_demo_fine.py`, image đang clone code từ GitHub:

```txt
git clone -b chonjohn/segmentation ... /app
```

Vì vậy sửa code local xong phải push lên GitHub trước khi deploy.

```bash
git add .
git commit -m "Fix Modal VoxTell setup"
git push origin chonjohn/segmentation
```

---

### Bước 6: Deploy Modal

```bash
PYTHONPATH="$(pwd)/.certfix" modal deploy modal_web_demo_fine.py
```

Sau khi deploy, Modal sẽ in ra URL dạng:

```txt
https://<account>--ai-medical-dept-web.modal.run
```

---

## 3. Lệnh chạy local

### Chạy backend production local

```bash
uvicorn serving.backend.server:app --host 0.0.0.0 --port 8000
```

Hoặc nếu dùng conda env:

```bash
conda run -n voxtell python -m uvicorn serving.backend.server:app --host 0.0.0.0 --port 8000
```

---

### Chạy frontend local

```bash
cd frontend
npm install
npm run dev
```

---

### Chạy full app bằng script

Nếu repo có `run.sh`:

```bash
chmod +x run.sh
./run.sh
```

---

## 4. Lệnh test sau khi deploy Modal

### Test health

```bash
curl -i "https://<modal-url>/api/health"
```

Kết quả mong muốn:

```json
{"status":"ok"}
```

hoặc:

```json
{"status":"ok","voxtell":"loaded"}
```

Tùy `server.py` hiện tại trả health như thế nào.

---

### Test qua frontend

Mở:

```txt
https://<modal-url>
```

Kiểm tra:

```txt
1. Load patient
2. CT volume hiện trong viewer
3. Chọn Multi / Axial / Coronal / Sagittal
4. Nhập prompt
5. Run Segmentation
6. Mask mới xuất hiện trong danh sách segmentations
7. Có thể segment nhiều prompt
8. Có thể hide/show/download/remove từng mask
```

---

## 5. Lệnh đổi `VOXTELL_MODAL_BASE_URL` ở backend production

Nếu backend production bên ngoài cần proxy sang Modal mới, đổi trong `.env` backend:

```env
VOXTELL_MODAL_BASE_URL=https://<new-modal-url>/api
```

Sau đó restart backend:

```bash
uvicorn serving.backend.server:app --host 0.0.0.0 --port 8000
```

Nếu dùng Docker:

```bash
docker compose up -d --build
```

---

## 6. Các file đã chỉnh / cần chú ý

```txt
modal_web_demo_fine.py
→ build Modal image
→ cài dependency
→ clone repo
→ chạy run.sh
→ expose Nginx port 8000

run.sh
→ start backend port 1711
→ start frontend port 2811

serving/backend/server.py
→ FastAPI backend chính
→ vẫn dùng file này cho Modal theo hướng hiện tại

frontend/src/components/CTViewer.tsx
→ khôi phục Multi view
→ khôi phục multi segmentation
→ mỗi mask có id riêng
```

---

## 7. Ghi nhớ lỗi thường gặp

### Lỗi 1: `ModuleNotFoundError: No module named 'asyncpg'`

Nguyên nhân:

```txt
Modal account mới build image lại từ đầu nên thiếu dependency.
```

Cách sửa:

```python
"bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install asyncpg'",
```

---

### Lỗi 2: Modal account mới không đọc được S3

Nguyên nhân:

```txt
Secret aws-s3 chưa được tạo trong account mới.
```

Cách sửa:

```bash
PYTHONPATH="$(pwd)/.certfix" modal secret create aws-s3 --from-dotenv .env --force
```

---

### Lỗi 3: Sửa code local nhưng deploy không đổi

Nguyên nhân:

```txt
modal_web_demo_fine.py clone code từ GitHub, không lấy code local.
```

Cách sửa:

```bash
git add .
git commit -m "Update Modal code"
git push origin chonjohn/segmentation
PYTHONPATH="$(pwd)/.certfix" modal deploy modal_web_demo_fine.py
```

---

### Lỗi 4: Segment mới ghi đè segment cũ

Nguyên nhân:

```txt
CTViewer dùng segmentation object đơn.
```

Cách sửa:

```txt
Dùng segmentations[].
Mỗi mask có id riêng.
NiiVue add/remove mask theo id.
```

---

## 8. Thứ tự chạy chuẩn từ đầu

```txt
1. modal token new
2. modal profile current
3. chuẩn bị .env đầy đủ S3/DB
4. modal secret create aws-s3 --from-dotenv .env --force
5. sửa modal_web_demo_fine.py nếu thiếu asyncpg
6. sửa CTViewer.tsx nếu cần multi segmentation
7. git add .
8. git commit
9. git push origin chonjohn/segmentation
10. PYTHONPATH="$(pwd)/.certfix" modal deploy modal_web_demo_fine.py
11. mở Modal URL
12. test load CT volume
13. test Multi view
14. test nhiều segmentation prompt
```

## 9. Lệnh chạy local:
Mở 2 terminal
Terminal 1:
```txt
cd ai-medical-dept
uvicorn serving.backend.server:app --host 0.0.0.0 --port 8000
```

Terminal 2:
```txt
cd ai-medical-dept/frontend
npm run dev
```