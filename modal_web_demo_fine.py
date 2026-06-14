import modal
import subprocess
import time

app = modal.App("ai-medical-dept-voxtell")

# Modal Secret phải chứa:
# S3_BUCKET_NAME, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
AWS_S3_SECRET_NAME = "aws-s3"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "nginx",
        "git",
        "curl",
        "ca-certificates",
        "wget",
        "bzip2",
        "dcm2niix",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .run_commands(
        # Cài Miniconda
        "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh",
        "bash /tmp/miniconda.sh -b -p /opt/conda",
        "rm /tmp/miniconda.sh",

        # Clone repo
        "git clone --depth=1 -b chonjohn/segmentation https://github.com/UngHoangLong/ai-medical-dept.git /app && cd /app && git rev-parse HEAD",

        # Accept Anaconda Terms of Service
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r'",

        # Tạo env voxtell
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda create -y -n voxtell python=3.12'",

        # Cài nodejs + nibabel bằng conda
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && conda install -c conda-forge nodejs -y'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && conda install -c conda-forge nibabel -y'",

        # Cài torch GPU
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126'",

        # Cài backend packages
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install pydicom'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install \"uvicorn[standard]\"'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install fastapi'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install nnunetv2'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install rt-utils'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install voxtell'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install python-multipart'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install boto3 python-dotenv'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install asyncpg'",

        # Cài local repo
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && cd /app && python -m pip install -e .'",

        # Download model
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && cd /app && python download_model.py'",

        # Patch frontend fallback URL: gọi backend qua /api thay vì localhost:1711
        # Code mới đã dùng VITE_* env; block này chỉ là fallback nếu còn URL hard-code trong repo.
        r"""cd /app && python - <<'PY'
from pathlib import Path
import re

for p in Path("/app/frontend/src").rglob("*"):
    if p.suffix not in {".ts", ".tsx", ".js", ".jsx"}:
        continue
    s = p.read_text(encoding="utf-8")
    original = s
    s = s.replace("http://localhost:1711", "/api")
    s = s.replace("http://127.0.0.1:1711", "/api")
    s = re.sub(r"https://[^'\"`]+modal\.run", "/api", s)
    if s != original:
        p.write_text(s, encoding="utf-8")
        print(f"Patched frontend API URL fallback in {p}")
PY""",

        # Frontend env: qua Nginx /api proxy về backend 1711 trong cùng container.
        # Modal Image.run_commands không hỗ trợ command bắt đầu trực tiếp bằng `cat`,
        # nên dùng Python để ghi file.
        r"""cd /app && python - <<'PY'
from pathlib import Path
p = Path('/app/frontend/.env.local')
p.write_text('VITE_BACKEND_URL=/api\nVITE_VOXTELL_API_BASE_URL=/api\n', encoding='utf-8')
print(p.read_text(encoding='utf-8'))
PY""",

        # Patch Vite: cho phép Modal domain
        r"""cd /app && python - <<'PY'
from pathlib import Path
import re

frontend = Path("/app/frontend")

for name in ["vite.config.ts", "vite.config.js", "vite.config.mts"]:
    p = frontend / name
    if not p.exists():
        continue

    s = p.read_text(encoding="utf-8")

    if "allowedHosts" in s:
        print(f"{p} already has allowedHosts")
        break

    # Nếu đã có server: { ... } thì thêm allowedHosts vào trong đó
    if re.search(r"server\s*:\s*\{", s):
        s = re.sub(
            r"server\s*:\s*\{",
            "server: {\n    allowedHosts: true,",
            s,
            count=1,
        )
    # Nếu chưa có server block thì thêm server block vào defineConfig({
    elif "defineConfig({" in s:
        s = s.replace(
            "defineConfig({",
            "defineConfig({\n  server: {\n    allowedHosts: true,\n  },",
            1,
        )
    else:
        print(f"Could not patch {p}: defineConfig not found")
        break

    p.write_text(s, encoding="utf-8")
    print(f"Patched {p} with server.allowedHosts = true")
    break
PY""",

        # Frontend deps
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && cd /app/frontend && npm install'",

        # Cho phép chạy run.sh
        "cd /app && chmod +x run.sh",
    )
)


def wait_for_port(port, timeout=180):
    import socket
    import time

    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(1)

    raise RuntimeError(f"Port {port} not ready after {timeout}s")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    max_containers=1,
    scaledown_window=20 * 60,
    secrets=[modal.Secret.from_name(AWS_S3_SECRET_NAME)],
)
@modal.concurrent(max_inputs=20)
@modal.web_server(8000, startup_timeout=600)
def web():
    nginx_conf = r"""
events {}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    access_log /dev/stdout;
    error_log /dev/stderr info;

    client_max_body_size 2G;

    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
    proxy_connect_timeout 3600s;
    client_body_timeout 3600s;
    send_timeout 3600s;

    server {
        listen 8000;
        server_name _;

        location / {
            proxy_pass http://127.0.0.1:2811;
            proxy_http_version 1.1;

            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";

            proxy_buffering off;
        }

        location /api/ {
            rewrite ^/api/(.*)$ /$1 break;
            proxy_pass http://127.0.0.1:1711;
            proxy_http_version 1.1;

            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

            proxy_buffering off;
            proxy_request_buffering off;
        }
    }
}
"""

    subprocess.run("mkdir -p /etc/nginx", shell=True, check=True)

    with open("/etc/nginx/nginx.conf", "w") as f:
        f.write(nginx_conf)

    subprocess.run("nginx -t", shell=True, check=True)

    subprocess.Popen(
        "cd /app && source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && ./run.sh",
        shell=True,
        executable="/bin/bash",
    )

    print("Waiting for backend 1711...")
    wait_for_port(1711, timeout=300)

    print("Waiting for frontend 2811...")
    wait_for_port(2811, timeout=180)

    subprocess.Popen(
        "nginx -g 'daemon off;'",
        shell=True,
    )

    print("Started nginx on port 8000")
    return