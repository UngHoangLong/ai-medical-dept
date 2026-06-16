-- ============================================================
-- AI Medical Dept — RDB migration (schema ai_demo, đã tồn tại)
--
-- Bảng ai_demo.patients / clinical_data / consultation_reports đã được tạo
-- sẵn (3 FK: clinical_data.patient_id, consultation_reports.patient_id,
-- consultation_reports.record_id). Migration này CHỈ bổ sung/đổi tên cột để
-- đủ dùng cho pipeline mới — không tạo lại từ đầu, không đổi FK hiện có.
--
-- Sau migration, S3 chỉ còn giữ FILE:
--   - dicom-raw/{pid}/{series_uid}.zip   → consultation_reports.dicom_s3_key
--   - ct-slices/{pid}/{series_uid}/*.png → consultation_reports.ct_slices_prefix
-- report_content (5 agent outputs) + clinical_data (raw + text) lưu trong DB.
--
-- Chạy: psql "$DATABASE_URL" -f db/schema.sql  (idempotent)
-- ============================================================

-- patients.full_name hiện NOT NULL nhưng clinical_data (NLST, de-identified)
-- không có tên thật bệnh nhân → cho phép NULL.
ALTER TABLE ai_demo.patients
    ALTER COLUMN full_name DROP NOT NULL;

-- clinical_text = output clinical_to_text(medical_history) — đoạn văn tiếng
-- Anh dùng trong mọi prompt + làm context cho chatbot.
ALTER TABLE ai_demo.clinical_data
    ADD COLUMN IF NOT EXISTS clinical_text TEXT;

-- Đổi tên cho đúng nội dung lưu (S3 *key*, không phải URL — presigned URL ký
-- lại mỗi lần cần, không lưu trong DB) + thêm series_uid (key lookup
-- pid+series_uid từ frontend), n_slices (cho CT Viewer), updated_at (cho
-- Patient History "mới nhất trước").
ALTER TABLE ai_demo.consultation_reports
    RENAME COLUMN ct_scan_raw_url TO dicom_s3_key;

ALTER TABLE ai_demo.consultation_reports
    RENAME COLUMN ct_scan_preprocessed_url TO ct_slices_prefix;

ALTER TABLE ai_demo.consultation_reports
    ADD COLUMN IF NOT EXISTS series_uid VARCHAR(255) UNIQUE,
    ADD COLUMN IF NOT EXISTS n_slices INTEGER,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE INDEX IF NOT EXISTS idx_consultation_reports_updated_at
    ON ai_demo.consultation_reports (updated_at DESC);

-- Lưu ý: cột `thread_id` (consultation_reports) giữ nguyên — dành cho phần
-- chatbot/pgvector của anh bạn, không đụng tới ở đây.
