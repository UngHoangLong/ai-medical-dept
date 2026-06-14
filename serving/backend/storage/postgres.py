"""
PgStorage — RDS Postgres storage cho clinical data + consultation reports.

Thay thế S3 `analysis-results/{pid}/{series_uid}.json`: report_content (5 agent
outputs) + clinical data (raw + clinical_text) được lưu trong schema `ai_demo`
(đã có sẵn patients/clinical_data/consultation_reports + FK — xem
db/schema.sql cho phần bổ sung). S3 chỉ còn giữ file (dicom-raw/, ct-slices/).
"""

import json
import os

import asyncpg

SCHEMA = "ai_demo"


class PgStorage:

    def __init__(self):
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            host=os.environ["DB_HOST"],
            port=int(os.environ.get("DB_PORT", "5432")),
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            database=os.environ.get("DB_NAME", "postgres"),
            min_size=1,
            max_size=5,
            ssl="require",
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    async def get_analysis(self, pid: str, series_uid: str) -> dict | None:
        """Load report_content đã lưu cho 1 (pid, series_uid). None nếu chưa có."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT report_id, report_content FROM {SCHEMA}.consultation_reports "
                "WHERE patient_id = $1 AND series_uid = $2",
                pid, series_uid,
            )
        if not row:
            return None

        report = json.loads(row["report_content"]) or {}
        report_id = str(row.get("report_id"))  # Lấy report_id từ DB để trả về cùng report_content (phục vụ thread_id cho FE)
        return {
            **report,
            "report_id": report_id
        }

    async def list_analyses(self) -> list[dict]:
        """Liệt kê tất cả bệnh nhân đã có report, mới nhất trước."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT patient_id, series_uid, updated_at FROM {SCHEMA}.consultation_reports "
                "WHERE series_uid IS NOT NULL ORDER BY updated_at DESC"
            )
        return [
            {
                "pid": row["patient_id"],
                "series_uid": row["series_uid"],
                "last_modified": row["updated_at"].isoformat(),
            }
            for row in rows
        ]

    async def save_analysis(
        self,
        pid: str,
        series_uid: str,
        clinical_data: dict,
        clinical_text: str,
        result: dict,
        dicom_s3_key: str,
        ct_slices_prefix: str,
        n_slices: int,
        gender: str | None = None,
    ) -> None:
        """Upsert patient + clinical_data (bản ghi mới) + consultation_reports."""
        medical_history_json = json.dumps(clinical_data, ensure_ascii=False)
        result_json = json.dumps(result, ensure_ascii=False)

        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                f"INSERT INTO {SCHEMA}.patients (patient_id, gender) VALUES ($1, $2) "
                "ON CONFLICT (patient_id) DO NOTHING",
                pid, gender,
            )

            record_id = await conn.fetchval(
                f"INSERT INTO {SCHEMA}.clinical_data (patient_id, medical_history, clinical_text) "
                "VALUES ($1, $2::jsonb, $3) RETURNING record_id",
                pid, medical_history_json, clinical_text,
            )

            await conn.execute(
                f"""
                INSERT INTO {SCHEMA}.consultation_reports
                    (patient_id, record_id, series_uid, dicom_s3_key,
                     ct_slices_prefix, n_slices, report_content, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, now())
                ON CONFLICT (series_uid) DO UPDATE SET
                    record_id        = EXCLUDED.record_id,
                    dicom_s3_key     = EXCLUDED.dicom_s3_key,
                    ct_slices_prefix = EXCLUDED.ct_slices_prefix,
                    n_slices         = EXCLUDED.n_slices,
                    report_content   = EXCLUDED.report_content,
                    updated_at       = now()
                """,
                pid, record_id, series_uid, dicom_s3_key,
                ct_slices_prefix, n_slices, result_json,
            )
