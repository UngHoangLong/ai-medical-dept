#!/bin/bash
# Tự động chạy build_npy_hub.py cho đến khi xong hết.
# Mỗi run bị giới hạn MAX_SECS giây — nếu stuck quá thì tự kill và restart.

KEYS=/Users/macbook/Documents/Demo_AI/data/subset_keys/test_npy_keys.txt
PID_MAP=/Users/macbook/Documents/Demo_AI/data/subset_keys/test_npy_pid_map.json
TMP=/tmp/npy_test_build
REPO=UngLong/openm3chest-npy-v2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAX_SECS=3600  # force-kill sau 1 giờ

RUN=0
while true; do
    RUN=$((RUN + 1))
    echo ""
    echo "=========================================="
    echo " Run #$RUN  —  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    # Chạy Python ở background
    python "$SCRIPT_DIR/build_npy_hub.py" \
        --keys-file     "$KEYS" \
        --pid-map       "$PID_MAP" \
        --tmp-dir       "$TMP" \
        --hf-repo       "$REPO" \
        --checkpoint    "$SCRIPT_DIR/npy_test_checkpoint.txt" \
        --workers       3 &
    PY_PID=$!

    # Killer: tự kill Python sau MAX_SECS
    ( sleep $MAX_SECS; kill -TERM $PY_PID 2>/dev/null ) &
    KILLER_PID=$!

    # Đợi Python xong
    wait $PY_PID
    EXIT=$?

    # Dừng killer nếu Python tự thoát trước
    kill $KILLER_PID 2>/dev/null
    wait $KILLER_PID 2>/dev/null

    # Dọn tmp
    rm -rf "$TMP/npy" "$TMP/_dicom"

    if [ $EXIT -eq 0 ]; then
        echo ""
        echo "Xong het! Hub: https://huggingface.co/datasets/$REPO"
        break
    fi

    echo ""
    echo "Exit $EXIT — restart sau 15s..."
    sleep 15
done
