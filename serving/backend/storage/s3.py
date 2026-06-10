"""
S3Storage — upload ảnh CT (PNG) lên S3 và tạo presigned URL để Modal tải về.

Chỉ ảnh đi qua S3 — prompt vẫn gửi thẳng trong payload tới Modal (nhỏ, gắn
với từng request, gửi trực tiếp rẻ hơn nhiều so với phải qua trung gian S3).
"""

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

import boto3
from botocore.exceptions import ClientError

PRESIGNED_URL_TTL = 3600  # giây — đủ cho 1 phiên phân tích trọn vẹn


def _object_key(cache_key: str, idx: int) -> str:
    # cache_key = "{pid}/{series_uid}"
    return f"ct-slices/{cache_key}/{idx:03d}.png"


def _analysis_key(cache_key: str) -> str:
    # cache_key = "{pid}/{series_uid}"
    return f"analysis-results/{cache_key}.json"


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

    def save_analysis(self, cache_key: str, result: dict) -> None:
        """Lưu kết quả phân tích (JSON) — chatbot /ask sẽ load lại làm context."""
        self.client.put_object(
            Bucket=self.bucket,
            Key=_analysis_key(cache_key),
            Body=json.dumps(result, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )

    def load_analysis(self, cache_key: str) -> dict | None:
        """Load lại kết quả phân tích đã lưu. Trả về None nếu chưa có."""
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=_analysis_key(cache_key))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise
        return json.loads(obj["Body"].read())
