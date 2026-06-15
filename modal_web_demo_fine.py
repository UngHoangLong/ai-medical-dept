import os
import subprocess
import time

import modal

APP_NAME = "ai-medical-dept-voxtell-s3-direct"
REPO_URL = "https://github.com/UngHoangLong/ai-medical-dept.git"
REPO_BRANCH = "chonjohn/segmentation"
REBUILD_MARKER = "S3_DIRECT_CLEAN_REBUILD_20260615_01"

# Load every required runtime/build variable from the local .env at deploy time.
# Do NOT put VOXTELL_MODAL_BASE_URL in .env for this app.
PROJECT_SECRET = modal.Secret.from_dotenv(__file__)

app = modal.App(APP_NAME)

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
        f"echo {REBUILD_MARKER}",

        # Miniconda
        "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh",
        "bash /tmp/miniconda.sh -b -p /opt/conda",
        "rm /tmp/miniconda.sh",

        # Clone the exact branch and print proof of the deployed commit/code.
        f"git clone --depth=1 -b {REPO_BRANCH} {REPO_URL} /app && cd /app && "
        "echo '=== MODAL GIT COMMIT ===' && git rev-parse HEAD && "
        "echo '=== FIND S3 DIRECT VOXTELL CODE ===' && "
        "grep -RIn --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=dist "
        "'modal_proxy\\|VOXTELL_MODAL_BASE_URL\\|S3_GET_START\\|DCM2NIIX_START\\|VOXTELL_PREDICT_START\\|VoxTellPredictor' "
        "serving/backend/server.py serving/backend/routes/voxtell.py || true",

        # Conda terms + env
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda create -y -n voxtell python=3.12'",

        # Runtime deps
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && conda install -c conda-forge nodejs nibabel -y'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && python -m pip install pydicom \"uvicorn[standard]\" fastapi nnunetv2 rt-utils voxtell python-multipart boto3 python-dotenv asyncpg'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && cd /app && python -m pip install -e .'",

        # VoxTell/MedGemma model download can require HF_TOKEN from .env.
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && cd /app && python download_model.py'",

        # Frontend: always call this same Modal container through nginx /api.
        r"""cd /app && python - <<'PY'
from pathlib import Path
import re

for p in Path('/app/frontend/src').rglob('*'):
    if p.suffix not in {'.ts', '.tsx', '.js', '.jsx'}:
        continue
    s = p.read_text(encoding='utf-8')
    original = s
    s = s.replace('http://localhost:1711', '/api')
    s = s.replace('http://127.0.0.1:1711', '/api')
    s = re.sub(r"https://[^'\"`]+modal\.run", "/api", s)
    if s != original:
        p.write_text(s, encoding='utf-8')
        print(f'Patched frontend API URL fallback in {p}')

p = Path('/app/frontend/.env.local')
p.write_text('VITE_BACKEND_URL=/api\nVITE_VOXTELL_API_BASE_URL=/api\n', encoding='utf-8')
print(p.read_text(encoding='utf-8'))
PY""",

        # Vite dev server allowed hosts for Modal domain.
        r"""cd /app && python - <<'PY'
from pathlib import Path
import re

frontend = Path('/app/frontend')
for name in ['vite.config.ts', 'vite.config.js', 'vite.config.mts']:
    p = frontend / name
    if not p.exists():
        continue
    s = p.read_text(encoding='utf-8')
    if 'allowedHosts' in s:
        print(f'{p} already has allowedHosts')
        break
    if re.search(r'server\s*:\s*\{', s):
        s = re.sub(r'server\s*:\s*\{', 'server: {\n    allowedHosts: true,', s, count=1)
    elif 'defineConfig({' in s:
        s = s.replace('defineConfig({', 'defineConfig({\n  server: {\n    allowedHosts: true,\n  },', 1)
    else:
        print(f'Could not patch {p}: defineConfig not found')
        break
    p.write_text(s, encoding='utf-8')
    print(f'Patched {p} with server.allowedHosts = true')
    break
PY""",

        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && cd /app/frontend && npm install'",
        "cd /app && chmod +x run.sh",
        secrets=[PROJECT_SECRET],
        force_build=True,
    )
)


def wait_for_port(port: int, timeout: int = 180) -> None:
    import socket

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
    secrets=[PROJECT_SECRET],
)
@modal.concurrent(max_inputs=20)
@modal.web_server(8000, startup_timeout=900)
def web():
    # Hard guard: this app must never proxy to old Modal.
    os.environ.pop("VOXTELL_MODAL_BASE_URL", None)

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
    with open("/etc/nginx/nginx.conf", "w", encoding="utf-8") as f:
        f.write(nginx_conf)
    subprocess.run("nginx -t", shell=True, check=True)

    print("APP_NAME =", APP_NAME)
    print("RUNTIME S3_BUCKET_NAME =", os.getenv("S3_BUCKET_NAME"))
    print("RUNTIME AWS_REGION =", os.getenv("AWS_REGION"))
    print("RUNTIME MODAL_PIPELINE_URL configured =", bool(os.getenv("MODAL_PIPELINE_URL")))
    print("RUNTIME DB_HOST configured =", bool(os.getenv("DB_HOST")))
    print("RUNTIME VOXTELL_MODAL_BASE_URL =", os.getenv("VOXTELL_MODAL_BASE_URL"))

    subprocess.Popen(
        "cd /app && source /opt/conda/etc/profile.d/conda.sh && conda activate voxtell && "
        "BACKEND_CONDA_ENV=voxtell BACKEND_PORT=1711 FRONTEND_PORT=2811 PUBLIC_BACKEND_URL=/api ./run.sh",
        shell=True,
        executable="/bin/bash",
    )

    print("Waiting for backend 1711...")
    wait_for_port(1711, timeout=600)

    print("Waiting for frontend 2811...")
    wait_for_port(2811, timeout=240)

    subprocess.Popen("nginx -g 'daemon off;'", shell=True)
    print("Started nginx on port 8000")
    return
