# Luồng chạy hệ thống sau khi refactor sang patient-based S3 DICOM flow

## 1. Mục tiêu refactor

Hệ thống đã được chỉnh từ flow **upload file thủ công** sang flow **tự động lấy dữ liệu theo bệnh nhân**.

### Trước đây

Frontend CTViewer cho phép:
- upload `.nii`
- upload `.nii.gz`
- upload `.zip` DICOM
- gọi `/convert`
- gọi `/predict` với `image + prompt`
- hỗ trợ multi segmentation

### Bây giờ

Frontend CTViewer:
- **không còn upload thủ công** `.nii/.nii.gz/.zip`
- **không còn dùng** `/convert`
- **không còn dùng** `sessionId`
- **không còn multi segmentation**
- tự nhận dữ liệu qua `pid` + `series_uid`
- tự gọi backend để lấy volume và chạy segmentation

---

## 2. Cấu trúc dữ liệu S3 đang dùng

```txt
s3://<bucket>/
├── ct-slices/{pid}/{series_uid}/000.png
└── dicom-raw/{pid}/{series_uid}.zip
```

Trong đó:
- `ct-slices/...` dùng cho viewer PNG slice (nếu cần ở luồng khác)
- `dicom-raw/{pid}/{series_uid}.zip` là dữ liệu gốc
- file `.zip` chứa các file `.dcm`

Backend hiện đang tự build key S3 theo quy tắc:

```txt
dicom-raw/{pid}/{series_uid}.zip
```

---

## 3. Luồng chạy tổng quát

```txt
Frontend có sẵn pid + series_uid
→ CTViewer nhận pid + series_uid từ App
→ CTViewer gọi GET /api/voxtell/volume/{pid}/{series_uid}
→ Backend tải dicom-raw/{pid}/{series_uid}.zip từ S3
→ Backend unzip các file .dcm
→ Backend convert tạm sang .nii.gz
→ Backend trả volume .nii.gz về frontend
→ NiiVue render volume 3D
→ User nhập prompt
→ CTViewer gọi POST /api/voxtell/predict
→ Backend lại lấy DICOM zip từ S3
→ unzip .dcm
→ convert tạm sang .nii.gz
→ chạy VoxTell segmentation
→ trả mask .nii.gz về frontend
→ Frontend overlay mask lên NiiVue
```

---

## 4. Thay đổi ở frontend

### 4.1. `App.tsx`

`App.tsx` đã có sẵn thông tin patient active trong session:
- `pid`
- `series_uid`

Việc cần làm là truyền các giá trị này xuống `CTViewer`.

### Trước đây

```tsx
<CTViewer status={status} />
```

### Sau refactor

```tsx
<CTViewer
  status={status}
  pid={active?.pid}
  seriesUid={active?.series_uid}
/>
```

### Ý nghĩa

CTViewer không còn quản lý file upload nữa, mà chỉ cần biết:
- đang xem bệnh nhân nào
- series nào cần load

---

### 4.2. `CTViewer.tsx`

CTViewer đã được refactor theo hướng patient-based.

#### Đã bỏ
- input upload `.nii`
- input upload `.nii.gz`
- input upload `.zip`
- gọi `/convert`
- `sessionId`
- cleanup DICOM session
- multi segmentation `segmentations[]`
- chế độ Multi view
- `Download All`

#### Còn lại / mới
- nhận `pid`
- nhận `seriesUid`
- khi có `pid + seriesUid` thì tự gọi endpoint volume
- load blob `.nii.gz` trả về từ backend vào NiiVue
- chỉ giữ **1 segmentation hiện tại**
- user nhập prompt rồi bấm Run Segmentation
- frontend gọi `/api/voxtell/predict`
- nhận mask `.nii.gz` và hiển thị overlay

#### Ý tưởng state mới

```ts
pid
seriesUid
volumeFile
prompt
segmentation
isLoadingVolume
isProcessing
error
```

---

## 5. Thay đổi ở backend

Backend không còn nhận file upload thủ công để segment nữa.

### 5.1. Endpoint mới lấy volume

```http
GET /voxtell/volume/{pid}/{series_uid}
```

### Chức năng

1. build S3 key:

```txt
dicom-raw/{pid}/{series_uid}.zip
```

2. tải zip từ S3
3. unzip DICOM ra thư mục tạm
4. tìm thư mục chứa `.dcm`
5. convert DICOM sang `.nii.gz` bằng `dcm2niix`
6. trả file `.nii.gz` về cho frontend

---

### 5.2. Endpoint segmentation mới

```http
POST /voxtell/predict
```

### Input

FormData:
- `pid`
- `series_uid`
- `prompt`

### Chức năng

1. build S3 key:

```txt
dicom-raw/{pid}/{series_uid}.zip
```

2. tải zip từ S3
3. unzip `.dcm`
4. convert sang `.nii.gz` tạm
5. đọc volume bằng `NibabelIOWithReorient`
6. chạy `VoxTellPredictor`
7. xuất segmentation mask `.nii.gz`
8. trả file `.nii.gz` về frontend

### Lưu ý

- segmentation **không lưu lên S3**
- mask chỉ trả về frontend
- RTSTRUCT/session upload flow cũ không còn là luồng chính

---

## 6. Cấu hình môi trường

### Backend / Modal cần

```env
S3_BUCKET_NAME=your-bucket-name
AWS_REGION=ap-southeast-2
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### Frontend cần

Khi chạy qua Modal + nginx proxy:

```env
VITE_BACKEND_URL=/api
VITE_VOXTELL_API_BASE_URL=/api
```

### Local frontend (nếu chạy local)

```env
VITE_BACKEND_URL=http://localhost:1711
VITE_VOXTELL_API_BASE_URL=http://localhost:1711
```

---

## 7. Luồng chạy trên Modal

File Modal đã được chỉnh để:

1. clone đúng repo / branch:

```txt
repo:  https://github.com/UngHoangLong/ai-medical-dept.git
branch: chonjohn/segmentation
```

2. cài môi trường backend/frontend
3. tải model VoxTell từ Hugging Face `mrokuss/VoxTell`
4. inject frontend env:

```env
VITE_BACKEND_URL=/api
VITE_VOXTELL_API_BASE_URL=/api
```

5. mount secret AWS S3 thông qua Modal Secret `aws-s3`
6. chạy `run.sh`
7. dùng nginx reverse proxy:
- `/` → frontend port `2811`
- `/api/` → backend port `1711`

---

## 8. Cách test sau khi deploy Modal

Giả sử app chạy ở URL:

```txt
https://dtdtchon123--ai-medical-dept-web-dev.modal.run
```

### 8.1. Test health

```bash
curl -i "https://dtdtchon123--ai-medical-dept-web-dev.modal.run/api/health"
```

Kết quả mong đợi:

```json
{"status":"ok","voxtell":"loaded"}
```

---

### 8.2. Test backend có đọc được dữ liệu từ S3 chưa

Ví dụ:
- `pid = 122833`
- `series_uid = 1.2.840.113654.2.55.307229896835702397101325832166669748691`

Chạy:

```bash
curl -L -o test_volume.nii.gz \
"https://dtdtchon123--ai-medical-dept-web-dev.modal.run/api/voxtell/volume/122833/1.2.840.113654.2.55.307229896835702397101325832166669748691"

ls -lh test_volume.nii.gz
gzip -t test_volume.nii.gz
```

### Ý nghĩa

Nếu file `.nii.gz` tải về thành công và `gzip -t` không lỗi thì chứng tỏ:

```txt
Modal backend → đọc được S3 → tải zip DICOM → unzip → convert → trả volume thành công
```

Trong lần test thực tế, file volume tải về có dung lượng khoảng **33 MB**, tức là luồng này đã hoạt động.

---

### 8.3. Test segmentation

```bash
curl -L -o test_seg.nii.gz \
  -X POST "https://dtdtchon123--ai-medical-dept-web-dev.modal.run/api/voxtell/predict" \
  -F "pid=122833" \
  -F "series_uid=1.2.840.113654.2.55.307229896835702397101325832166669748691" \
  -F "prompt=lung nodule"

ls -lh test_seg.nii.gz
gzip -t test_seg.nii.gz
```

Nếu `test_seg.nii.gz` tải về được và hợp lệ thì chứng tỏ:

```txt
S3 DICOM zip → convert volume → VoxTell predict → trả mask thành công
```

Trong lần test thực tế, file mask có dung lượng khoảng **158 KB**, đây là bình thường vì segmentation mask thường nhỏ hơn volume rất nhiều.

---

## 9. Kết quả test thực tế đã xác nhận

### Health

```txt
GET /api/health → OK
{"status":"ok","voxtell":"loaded"}
```

### Volume

```txt
GET /api/voxtell/volume/{pid}/{series_uid} → OK
Tải được file test_volume.nii.gz ~33 MB
gzip -t pass
```

### Segmentation

```txt
POST /api/voxtell/predict → OK
Tải được file test_seg.nii.gz ~158 KB
gzip -t pass
```

=> Kết luận:

```txt
Flow patient-based S3 DICOM → volume → segmentation đã chạy thành công.
```

---

## 10. Tóm tắt nhanh

### Những gì đã đạt được

- bỏ hoàn toàn upload thủ công `.nii/.nii.gz/.zip` ở CTViewer
- CTViewer tự load volume theo `pid + series_uid`
- backend tự lấy DICOM zip từ S3
- backend tự convert tạm sang `.nii.gz`
- NiiVue vẫn dùng để hiển thị 3D volume
- segmentation chạy theo `pid + series_uid + prompt`
- mask trả về frontend
- Modal đã chạy được full flow
- test thực tế đã xác nhận volume và segmentation đều OK

### Flow cuối cùng

```txt
Frontend cache pid + series_uid
→ CTViewer tự load volume qua /api/voxtell/volume/{pid}/{series_uid}
→ Backend lấy DICOM zip từ S3 và convert
→ NiiVue hiển thị volume
→ User nhập prompt
→ Frontend gọi /api/voxtell/predict
→ Backend chạy VoxTell
→ Trả mask .nii.gz về frontend
```
