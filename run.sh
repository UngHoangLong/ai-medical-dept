#!/usr/bin/env bash
# run.sh — Start AI Medical Department backend + frontend with VoxTell CT Viewer
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-1711}"
FRONTEND_PORT="${FRONTEND_PORT:-2811}"
BACKEND_CONDA_ENV="${BACKEND_CONDA_ENV:-${CONDA_DEFAULT_ENV:-ai-chatbot-pro}}"
BACKEND_URL="http://localhost:${BACKEND_PORT}"
FRONTEND_URL="http://localhost:${FRONTEND_PORT}"

check_prerequisites() {
    local errors=0

    if ! command -v conda &>/dev/null; then
        for p in "$HOME/miniconda3/bin" "$HOME/anaconda3/bin" "/c/Users/$USER/miniconda3/Scripts" "/c/Users/$USER/anaconda3/Scripts"; do
            [ -f "$p/conda" ] && export PATH="$p:$PATH" && break
        done
        if ! command -v conda &>/dev/null; then
            echo -e "${RED}Error:${RESET} conda not found. Run this script in Anaconda Prompt/Git Bash after conda init." >&2
            errors=$((errors + 1))
        fi
    fi

    if command -v conda &>/dev/null; then
        if ! conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$BACKEND_CONDA_ENV"; then
            echo -e "${RED}Error:${RESET} conda environment '$BACKEND_CONDA_ENV' not found." >&2
            echo "  Current default is BACKEND_CONDA_ENV=${BACKEND_CONDA_ENV}" >&2
            echo "  To use another env: BACKEND_CONDA_ENV=voxtell ./run.sh" >&2
            errors=$((errors + 1))
        fi
    fi

    if ! command -v npm &>/dev/null; then
        echo -e "${RED}Error:${RESET} npm not found. Install Node.js 20.x or higher." >&2
        errors=$((errors + 1))
    elif command -v node &>/dev/null; then
        node_major=$(node --version 2>/dev/null | sed 's/v\([0-9]*\).*/\1/')
        if [ -n "$node_major" ] && [ "$node_major" -lt 20 ]; then
            echo -e "${RED}Error:${RESET} Node.js v${node_major} is too old — version 20.x or higher is required." >&2
            errors=$((errors + 1))
        fi
    fi

    if [ ! -f "$SCRIPT_DIR/frontend/package.json" ]; then
        echo -e "${RED}Error:${RESET} frontend/package.json not found. Run from repo root." >&2
        errors=$((errors + 1))
    elif [ ! -f "$SCRIPT_DIR/frontend/node_modules/.bin/vite" ]; then
        echo -e "${RED}Error:${RESET} Frontend dependencies not installed." >&2
        echo "  Run: cd frontend && npm install" >&2
        errors=$((errors + 1))
    fi

    if [ ! -f "$SCRIPT_DIR/serving/backend/server.py" ]; then
        echo -e "${RED}Error:${RESET} serving/backend/server.py not found." >&2
        errors=$((errors + 1))
    fi

    if [ ! -d "$SCRIPT_DIR/models/voxtell_v1.1" ]; then
        echo -e "${RED}Error:${RESET} VoxTell model not found at models/voxtell_v1.1/." >&2
        echo "  Put/download the VoxTell model into: $SCRIPT_DIR/models/voxtell_v1.1" >&2
        errors=$((errors + 1))
    fi

    if command -v conda &>/dev/null && conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$BACKEND_CONDA_ENV"; then
        if ! conda run -n "$BACKEND_CONDA_ENV" dcm2niix -h >/dev/null 2>&1; then
            echo -e "${RED}Error:${RESET} dcm2niix not found inside env '$BACKEND_CONDA_ENV'." >&2
            echo "  Install: conda install -n $BACKEND_CONDA_ENV -c conda-forge dcm2niix -y" >&2
            errors=$((errors + 1))
        fi
    fi

    if [ "$errors" -gt 0 ]; then exit 1; fi
}

upsert_frontend_env() {
    local key="$1"
    local value="$2"
    local env_file="$SCRIPT_DIR/frontend/.env.local"
    touch "$env_file"
    if grep -q "^${key}=" "$env_file"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$env_file"
    else
        printf '\n%s=%s\n' "$key" "$value" >> "$env_file"
    fi
}

BACKEND_PID=""; FRONTEND_PID=""
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${RESET}"
    for pid in "$FRONTEND_PID" "$BACKEND_PID"; do
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "$FRONTEND_PID" "$BACKEND_PID"; do
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    echo -e "${GREEN}Done.${RESET}"
}
trap cleanup EXIT INT TERM

check_prerequisites

upsert_frontend_env "VITE_BACKEND_URL" "$BACKEND_URL"
upsert_frontend_env "VITE_VOXTELL_API_BASE_URL" "$BACKEND_URL"

echo ""
echo -e "${BOLD}AI Medical Department + VoxTell CT Viewer${RESET}"
echo -e "─────────────────────────────────────────"
echo -e "  Backend env: ${GREEN}${BACKEND_CONDA_ENV}${RESET}"
echo -e "  Backend URL: ${GREEN}${BACKEND_URL}${RESET}"
echo -e "  Frontend URL: ${GREEN}${FRONTEND_URL}${RESET}"
echo ""

echo -e "${CYAN}Starting backend...${RESET}"
echo -e "  ${YELLOW}Note:${RESET} VoxTell model loading may take 30–60s."
(
  cd "$SCRIPT_DIR"
  conda run --no-capture-output -n "$BACKEND_CONDA_ENV" \
    python -m uvicorn serving.backend.voxtell_modal_server:app \
    --host 0.0.0.0 --port "$BACKEND_PORT"
) &
BACKEND_PID=$!

sleep 4

echo -e "${CYAN}Starting frontend...${RESET}"
(
  cd "$SCRIPT_DIR/frontend"
  npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" --strictPort 2>&1
) &
FRONTEND_PID=$!

echo ""
echo -e "  Backend:  ${GREEN}${BACKEND_URL}${RESET}"
echo -e "  Frontend: ${GREEN}${FRONTEND_URL}${RESET}  ${BOLD}← open this${RESET}"
echo ""
echo -e "  Press ${BOLD}Ctrl+C${RESET} to stop both servers."
echo -e "─────────────────────────────────────────"
echo ""

wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
