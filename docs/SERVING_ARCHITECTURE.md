# Serving Architecture — Backend, Modal, Frontend, S3

Tài liệu này mô tả luồng hệ thống **hiện tại** (production) của phần serving:
backend FastAPI (VPS), GPU inference (Modal), frontend (Vercel), và S3 storage.
Mục tiêu: đủ context để team code tiếp 2 chức năng còn thiếu — **Chatbot (`/ask`)**
và **CT Viewer**.

**Frontend (luôn cập nhật theo nhánh `dev`)**: https://ai-medical-dept.vercel.app/

---

## 1. Tổng quan luồng end-to-end

```
Bác sĩ (frontend)
  │  POST /api/v1/analyze
  │  FormData: pid, series_uid, clinical_data (JSON string), dicom_zip (.zip)
  ▼
FastAPI backend (VPS, Docker)
  │
  ├─ 1. cache_key = "{pid}/{series_uid}"
  │     → check S3 analysis-results/{cache_key}.json — nếu có sẵn, trả về luôn
  │
  ├─ 2. Unzip .dcm → HU volume → 85 slices RGB (HU windowing 3-channel)
  │     → upload PNG lên S3 ct-slices/{cache_key}/{idx:03d}.png
  │     → tạo presigned URL (TTL 1h) cho từng slice
  │
  ├─ 3. Build 5 prompt (formatter, KHÔNG chạy trên Modal):
  │     screening, detail(*), cardiology, oncology, finding_impression
  │     (*) detail chỉ chạy nếu screening.nodule_presence == "Yes" (run_if)
  │
  ├─ 4. POST Modal /infer_batch (1 request — toàn bộ 5 task)
  │     Modal tải ảnh từ S3 1 lần (RAM cache theo cache_key), chạy lần lượt
  │     5 task qua vLLM + LoRA tương ứng, share KV prefix cache (ảnh CT)
  │
  ├─ 5. Build prompt Agent 5 (Verification — MedGemma base, KHÔNG LoRA)
  │     nhúng: clinical_text + 4 report của agent trên + ảnh CT
  │     → POST Modal /infer_batch lần 2 (ảnh đã có sẵn trong RAM cache Modal)
  │
  ├─ 6. Highlight findings/impression/verification (LLM text-only, hiện tại
  │     DeepSeek) → chèn <hl c="critical|warning|normal"> quanh cụm từ gốc
  │
  ├─ 7. Lưu kết quả → S3 analysis-results/{cache_key}.json
  │
  ▼
Trả AnalyzeResponse (JSON) → frontend hiển thị (AgentResults, ChatPanel, CTViewer)
```

**CD**: push/merge vào `dev` → GitHub Actions SSH vào VPS → `git pull` + `docker
compose up -d --build` (chỉ backend). Frontend deploy qua Vercel khi merge `dev`.

---

## 2. Cấu trúc file & vai trò

### 2.1 Backend (`serving/backend/`)

| File | Vai trò |
|---|---|
| `server.py` | Khởi tạo FastAPI app, CORS, mount router `/api/v1`, tạo `MedicalPipeline` ở `startup`. |
| `pipeline.py` | **Orchestrator chính** — `MedicalPipeline.run()` thực hiện toàn bộ luồng ở mục 1. Cũng có `get_analysis()` (load cache) và `list_analyses()` (list lịch sử). |
| `modal_client.py` | HTTP client gọi 2 endpoint Modal: `call_modal()` (1 task / `/infer`) và `call_modal_pipeline()` (nhiều task / `/infer_batch`). |
| `highlighter_client.py` | Gọi LLM text-only (provider config qua `.env`: Groq/Gemini/Anthropic/OpenAI/OpenRouter/DeepSeek) để lấy danh sách cụm từ cần highlight. Backend tự chèn tag `<hl>` — model KHÔNG được sửa text gốc. Lỗi → fallback trả nguyên text. |
| `storage/s3.py` | Tất cả thao tác S3: upload slices, presigned URL, save/load/list kết quả phân tích. Xem mục 3. |
| `routes/analyze.py` | Định nghĩa endpoint REST (mục 4). |
| `schemas.py` | Pydantic models cho request/response. |

### 2.2 Agents / formatters (`agents/`)

Mỗi agent (trừ `highlighter`, `verification`) có 1 LoRA adapter riêng (host trên
HuggingFace, load vào Modal). `*/formatter.py` build prompt (inference) và parse
output JSON/text — đây là nơi **build_prompt và parse_output sống**, KHÔNG nằm
trong pipeline.py.

| Agent | Input | Output | Format |
|---|---|---|---|
| `radiology` (`screening` + `detail`) | ảnh CT + clinical text | 8 trường chest_abn_* + nodule_presence (screening); 4 trường nodule_* (detail, conditional) | JSON |
| `cardiology` | ảnh CT + clinical text | `CVD_diagnosis`, `CVD_mortality` | JSON |
| `oncology` | ảnh CT + clinical text | `lung_cancer_risk` | JSON |
| `finding_impression` | ảnh CT (KHÔNG nhận clinical text — prompt cố định, khớp dataset train) | `findings`, `impression` (free-text narrative) | regex section split |
| `verification` (Agent 5) | ảnh CT + clinical text + 4 report trên (radiology/cardiology/oncology/finding_impression) | free-text — AGREE/DISAGREE từng điểm | `<unused94>thought` block (MedGemma base CoT) |
| `highlighter` | text thuần (findings/impression/verification) | list `{phrase, category}` | JSON, backend tự `apply_highlights()` |

`agents/data_prep/`:
- `clinical_text.py` — `clinical_to_text(clinical_data)`: convert JSON clinical
  (NLST schema: `demo`, `smoking`, `disease_his`, `cancer_his`, `fam_lc`) →
  đoạn văn tiếng Anh dùng trong mọi prompt.
- `ct_processor.py` — `dicom_dir_to_slices()`: .dcm → HU volume → 85 PIL RGB
  slices (3-channel HU window: bone/lung, soft tissue, brain).

### 2.3 Modal (`serving/modal/app.py`)

1 app `ai-medical-dept`, 1 class `MedicalInference` (A100-80GB, vLLM, max 1
container, `scaledown_window=300s`).

- **RAM cache `_image_cache: dict[cache_key → list[PIL.Image]]`** — ảnh CT chỉ
  tải từ S3 1 lần/scan, dùng lại cho mọi agent + Agent 5 + (sau này) chatbot
  trong cùng phiên. **Mất khi container scale-down sau 5 phút idle** — gọi lại
  cần gửi `image_urls` (presigned, TTL 1h) để tải lại.
- `/infer` — 1 task, dùng cho test/standalone.
- `/infer_batch` — N task tuần tự trong 1 request, hỗ trợ `run_if` (điều kiện
  chạy dựa trên output JSON của task trước) — dùng cho `/analyze`.
- `lora_request=None` (`adapter: "base"`) → MedGemma base, không LoRA — dùng
  cho Agent 5 verification.

### 2.4 Frontend (`frontend/src/`)

| File | Vai trò |
|---|---|
| `App.tsx` | State chính: danh sách `patients` (session, lưu `sessionStorage` qua `lib/patientStore.ts`), `activeId`, gọi `/analyze` và `/analysis/{pid}/{series_uid}`. |
| `components/PatientModal.tsx` | Form upload bệnh nhân mới (.json clinical + .zip DICOM) → `/analyze`. |
| `components/PatientHistoryModal.tsx` | **(mới)** Gọi `/api/v1/analyses`, hiển thị danh sách bệnh nhân đã phân tích, click → `/api/v1/analysis/{pid}/{series_uid}` để load lại. |
| `components/PatientTabs.tsx` | Tab chuyển bệnh nhân trong session. |
| `components/AgentResults.tsx` | Hiển thị kết quả 5 agent — render `<hl c="...">` + `**bold**` markdown, group verification theo `**Label**:`. |
| `components/CTViewer.tsx` | **Placeholder** — "Viewer coming soon". Cần làm. |
| `components/ChatPanel.tsx` | UI chat đã xong, gọi `POST /api/v1/ask` — **endpoint backend CHƯA tồn tại**. |
| `lib/patientStore.ts` | Persist `patients[]` + `activeId` vào `sessionStorage` (mất khi đóng tab — đây là lý do cần Patient History để load lại từ S3). |
| `types/api.ts` | TypeScript types khớp `schemas.py`. |

---

## 3. S3 — đang lưu gì

Bucket: `S3_BUCKET_NAME` (region `AWS_REGION`), client trong `storage/s3.py`.

```
s3://<bucket>/
├── ct-slices/{pid}/{series_uid}/{000..084}.png
│     85 (hoặc ít hơn) ảnh PNG RGB, 3-channel HU-windowed, lossless.
│     Upload 1 lần/scan (key cache `_uploaded` trong S3Storage tránh re-upload
│     trong cùng phiên backend). KHÔNG có TTL — tồn tại vĩnh viễn cho tới khi
│     bị xoá thủ công.
│
└── analysis-results/{pid}/{series_uid}.json
      Toàn bộ AnalyzeResponse (radiology/cardiology/oncology/
      finding_impression/verification, đã highlight). Đây là cache chính:
      nếu file này tồn tại, /analyze trả về NGAY, không gọi lại Modal.
```

**Quan trọng cho chatbot**: `analysis-results/{...}.json` hiện **KHÔNG lưu
`clinical_data`** — chỉ lưu output. Nếu `/ask` cần clinical context (đặc biệt
khi load lại từ Patient History, lúc đó `clinical_data` gốc không còn ở
frontend), cần bổ sung lưu `clinical_data` vào file này (hoặc 1 file riêng
`analysis-results/{pid}/{series_uid}_clinical.json`).

**Presigned URL**: `presigned_urls()` ký lại mỗi lần gọi (TTL 3600s) dựa theo
key cố định `ct-slices/{cache_key}/{idx:03d}.png` — không cần lưu URL, chỉ cần
biết `cache_key` + số lượng slice.

---

## 4. API hiện có (`/api/v1`)

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/analyze` | Chạy full pipeline (hoặc trả cache nếu đã có). FormData: `pid`, `series_uid`, `clinical_data`, `dicom_zip`. |
| `GET` | `/analysis/{pid}/{series_uid}` | Load lại kết quả đã lưu trên S3. 404 nếu chưa có. |
| `GET` | `/analyses` | **(mới)** List `{pid, series_uid, last_modified}[]` — toàn bộ bệnh nhân đã phân tích, mới nhất trước. |

---

## 5. Việc cần làm — Chatbot (`POST /api/v1/ask`)

Frontend (`ChatPanel.tsx`) đã gửi sẵn:
```jsonc
{ "cache_key": "pid/series_uid", "message": "...", "history": [...], "context": <AnalyzeResponse> }
```

**Cần làm ở backend:**
1. Route mới `serving/backend/routes/chat.py`, schema `AskRequest`/`AskResponse`.
2. Đề xuất 2 cấp độ:
   - **Câu hỏi chung** (giải thích report, so sánh agent, follow-up...) → gọi
     LLM text-only (giống `highlighter_client.py` — Groq/Gemini/...) với
     context = `result` JSON (đã có trong request `context`) + `clinical_text`
     (cần `clinical_to_text()` — xem điểm về việc thiếu `clinical_data` ở mục 3).
     KHÔNG cần Modal/GPU → nhanh, rẻ.
   - **Câu hỏi cần "nhìn lại ảnh CT"** (vd "vùng nào trong ảnh cho thấy nodule?")
     → gọi Modal `/infer` với `adapter: "base"`, `cache_key`, kèm `image_urls`
     (presigned mới — `storage.presigned_urls(cache_key, n_slices)`; cần biết
     `n_slices`, có thể lưu kèm trong `analysis-results` hoặc đếm qua
     `list_objects_v2` prefix `ct-slices/{cache_key}/`). Lưu ý: RAM cache ảnh
     trên Modal có thể đã bị evict (sau 5 phút idle) → luôn gửi `image_urls`
     để fallback tải lại an toàn.
3. Lưu lịch sử chat? (tuỳ — hiện chưa có yêu cầu lưu).

---

## 6. Việc cần làm — CT Viewer

`CTViewer.tsx` hiện là placeholder, đã có sẵn UI prev/next slice (`◄ — / 85 ►`).

**Cần làm ở backend:**
- Endpoint mới (vd `GET /api/v1/ct-slices/{pid}/{series_uid}`) trả về
  `list[str]` presigned URL — gọi `storage.presigned_urls(cache_key, n_slices)`.
  Cần `n_slices`: hoặc lưu số lượng slice vào `analysis-results/{...}.json` lúc
  `pipeline.run()` (đơn giản nhất — thêm 1 field), hoặc đếm bằng
  `list_objects_v2(prefix="ct-slices/{cache_key}/")`.

**Cần làm ở frontend:**
- `CTViewer` gọi endpoint trên khi `status === 'done'`, lưu mảng URL, render
  `<img src={urls[currentIdx]}>`, nút `◄ ►` đổi `currentIdx` (đã có sẵn UI).
- Presigned URL hết hạn sau 1h — nếu cần xem lâu, gọi lại endpoint để refresh.

---

## 7. Lưu ý / gotchas

- **Non-determinism Agent 5**: vLLM continuous batching → cùng input có thể ra
  output khác nhau giữa các lần chạy dù `temperature=0`. Frontend đã thiết kế
  graceful-degradation (`AgentResults.tsx`: `splitVerificationSections`,
  `splitLabeledGroups`, `splitNumberedList`) để hiển thị tốt dù format thay đổi.
- **Highlighter** chỉ chèn tag quanh substring khớp CHÍNH XÁC — không bao giờ
  sửa/thêm text gốc. An toàn để gọi external LLM API.
- **Modal cold start** ~6 phút (vLLM load). `warmup()` trong `app.py` có thể
  gọi trước để giảm độ trễ.
- **CD**: chỉ backend tự deploy (GitHub Actions → SSH → `docker compose up -d
  --build` trên VPS). Frontend qua Vercel auto-deploy từ `dev`.
