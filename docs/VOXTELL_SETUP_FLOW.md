# Luồng tổng VoxTell Web + Auto Segmentation

## 1. Kiến trúc tổng thể

```text
User mở web
→ Nginx expose port 8000
→ Frontend Vite chạy port 2811
→ Backend FastAPI chạy port 1711
→ server.py xử lý API
→ VoxTell service chạy trên Modal/GPU
```

Khi chạy local production backend bên ngoài Modal:

```text
Frontend/Local backend
→ proxy request sang VOXTELL_MODAL_BASE_URL
→ Modal lấy dữ liệu từ S3
→ Modal chạy VoxTell bằng GPU
→ trả CT volume/mask về frontend
```

---

## 2. Nguồn dữ liệu

```text
Report/analysis:
Postgres
→ backend gọi pipeline.get_analysis(pid, series_uid)
→ LLM đọc report để chọn Top-3 finding quan trọng

DICOM gốc:
S3
→ dicom-raw/{pid}/{series_uid}.zip
→ Modal tải DICOM ZIP
→ convert DICOM sang NIfTI
→ đưa vào VoxTell

CT slices PNG:
Không phải input chính cho VoxTell segmentation.
```

---

## 3. Luồng load CT volume

```text
App.tsx truyền pid + series_uid vào CTViewer.tsx
→ CTViewer gọi GET /api/v1/voxtell/volume/{pid}/{series_uid}
→ backend proxy sang Modal
→ Modal lấy DICOM ZIP từ S3
→ convert DICOM → NIfTI
→ trả .nii.gz về frontend
→ NiiVue hiển thị CT volume
```

---

## 4. Luồng auto segmentation mới

```text
CT volume load xong
→ CTViewer tự gọi POST /api/v1/voxtell/predict-auto
→ backend lấy analysis từ Postgres
→ LLM chọn Top-3 finding quan trọng
→ backend tạo prompt candidates
→ VoxTell thử từng prompt
→ backend đọc mask .nii.gz và kiểm tra nonzero
→ chọn mask tốt nhất
→ trả mask + metadata về frontend
→ frontend overlay auto mask
```

Ví dụ:

```text
Top-3 từ LLM:
1. smooth 4~6mm solid nodule in right lower lobe
2. emphysematous changes with peripheral reticulation
3. 3mm nonspecific nodule in left lower lobe

Prompt candidates:
1. lung nodule
2. pulmonary nodule
3. right lower lobe lung nodule
4. solid pulmonary nodule
5. smooth 4~6mm solid nodule in right lower lobe

Backend chọn prompt có mask hợp lý, ví dụ:
lung nodule
```

---

## 5. Manual segmentation vẫn giữ nguyên

```text
Bác sĩ nhập prompt thủ công
→ POST /api/v1/voxtell/predict
→ backend proxy sang Modal
→ VoxTell segment theo prompt bác sĩ
→ trả mask .nii.gz
→ frontend overlay thêm mask manual
```

Auto segmentation chỉ là gợi ý ban đầu từ report. Bác sĩ vẫn có thể segment thêm vùng khác bằng manual prompt.

---

## 6. API chính

```text
GET  /api/v1/voxtell/volume/{pid}/{series_uid}
POST /api/v1/voxtell/prompt-auto
POST /api/v1/voxtell/predict-auto
POST /api/v1/voxtell/predict
GET  /api/health
```

---

## 7. Metadata frontend nhận từ auto segmentation

Backend trả mask `.nii.gz` kèm headers:

```text
x-voxtell-prompt          prompt được chọn
x-voxtell-source          llm hoặc rule_fallback
x-voxtell-top3            Top-3 finding từ report
x-voxtell-candidate-logs  log các prompt đã thử và nonzero voxels
```

Frontend hiển thị:

```text
Top-3 selected by LLM
Selected VoxTell prompt
Prompt candidates
nonzero voxels
```

---

## 8. Setup Modal account mới

```text
1. modal token new
2. modal profile current
3. tạo lại Modal secret aws-s3 từ .env
4. bảo đảm .env có S3 + DB/Postgres variables
5. cài đủ dependency như boto3, python-dotenv, asyncpg
6. commit và push code lên GitHub
7. deploy bằng modal deploy modal_web_demo_fine.py
8. test /api/health
9. mở Modal URL và test load CT + segmentation
```

Lưu ý quan trọng:

```text
Modal Secret theo từng account/workspace nên đổi Modal account là phải tạo lại secret.
Nếu modal_web_demo_fine.py clone code từ GitHub thì sửa local xong phải git push trước khi deploy.
```

---

## 9. Test nhanh

```bash
curl -i "https://<modal-url>/api/health"
```

```bash
curl -X POST "http://localhost:8000/api/v1/voxtell/predict-auto" \
  -F "pid=122833" \
  -F "series_uid=1.2.840.113654.2.55.307229896835702397101325832166669748691" \
  --output auto_voxtell_seg.nii.gz
```

Kiểm tra mask:

```text
max > 0
nonzero voxels > 0
```

---

## 10. Kết luận

Luồng tổng hiện tại là:

```text
Frontend load bệnh nhân
→ backend/Modal lấy DICOM từ S3 để hiển thị CT
→ backend lấy report từ Postgres
→ LLM chọn Top-3 finding
→ backend tạo nhiều prompt candidates
→ VoxTell chạy trên Modal GPU
→ backend chọn mask tốt nhất
→ frontend hiển thị auto mask
→ bác sĩ có thể chạy manual segmentation tiếp
```

---

## 11. Lệnh chạy local:
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