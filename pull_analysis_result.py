import os
import json
import boto3
from dotenv import load_dotenv

# Load file .env ở cùng thư mục đang chạy script
load_dotenv("/d/Demo-Namki/ai-medical-dept/.env")

PID = "122833"
SERIES_UID = "1.2.840.113654.2.55.307229896835702397101325832166669748691"

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")

if not S3_BUCKET_NAME:
    raise RuntimeError("Missing S3_BUCKET_NAME in .env")

key = f"analysis-results/{PID}/{SERIES_UID}.json"

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
)

print(f"Downloading s3://{S3_BUCKET_NAME}/{key}")

obj = s3.get_object(
    Bucket=S3_BUCKET_NAME,
    Key=key,
)

data = json.loads(obj["Body"].read().decode("utf-8"))

with open("analysis-result.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Saved to analysis-result.json")