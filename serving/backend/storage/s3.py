"""
S3Storage — upload ảnh CT (PNG) + DICOM gốc lên S3 và tạo presigned URL để
Modal/CT Viewer tải về.

Kết quả phân tích (report_content) + clinical_data KHÔNG còn lưu ở S3 — xem
serving/backend/storage/postgres.py (PgStorage, db/schema.sql). S3 chỉ giữ
file:
  - dicom-raw/{pid}/{series_uid}.zip   (DICOM gốc, cho viewer)
  - ct-slices/{pid}/{series_uid}/*.png (PNG đã windowing, cho Modal + viewer)
"""

import io
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

import boto3

PRESIGNED_URL_TTL = 3600  # giây — đủ cho 1 phiên phân tích trọn vẹn


def _object_key(cache_key: str, idx: int) -> str:
    # cache_key = "{pid}/{series_uid}"
    return f"ct-slices/{cache_key}/{idx:03d}.png"


def _dicom_zip_key(cache_key: str) -> str:
    # cache_key = "{pid}/{series_uid}"
    return f"dicom-raw/{cache_key}.zip"


class S3Storage:

    def __init__(self):
        self.bucket = os.environ["S3_BUCKET_NAME"]
        # boto3 tự đọc AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY từ env
        self.client = boto3.client("s3", region_name=os.environ["AWS_REGION"])

    def upload_images(self, cache_key: str, images: list[Image.Image]) -> int:
        """Upload từng ảnh (PNG, lossless) lên S3 song song. Trả về số lượng đã upload."""
        def _put(item: tuple[int, Image.Image]) -> None:
            idx, img = item
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            self.client.put_object(
                Bucket=self.bucket,
                Key=_object_key(cache_key, idx),
                Body=buf.getvalue(),
                ContentType="image/png",
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_put, enumerate(images)))
        return len(images)

    def upload_dicom_zip(self, cache_key: str, zip_bytes: bytes) -> str:
        """Upload nguyên file .zip DICOM gốc lên S3. Trả về S3 key đã lưu."""
        key = _dicom_zip_key(cache_key)
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=zip_bytes,
            ContentType="application/zip",
        )
        return key

    def presigned_urls(self, cache_key: str, n_images: int) -> list[str]:
        """Ký lại URL cho n_images đầu tiên (theo idx) của 1 cache_key — rẻ, không cần re-upload."""
        return [
            self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": _object_key(cache_key, i)},
                ExpiresIn=PRESIGNED_URL_TTL,
            )
            for i in range(n_images)
        ]
