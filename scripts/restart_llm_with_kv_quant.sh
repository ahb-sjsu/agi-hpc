#!/bin/bash
# Restart the Qwen/Gemma llama-server with KV cache quantization.
#
# Saves ~1.6 GB VRAM by using q8_0 keys + q4_0 values.
# Keys use q8_0 (higher precision for attention routing).
# Values use q4_0 (more tolerant of quantization).
#
# Usage:
#   bash scripts/restart_llm_with_kv_quant.sh           # restart spock (GPU 0)
#   bash scripts/restart_llm_with_kv_quant.sh --kirk     # restart kirk (GPU 1)
#   bash scripts/restart_llm_with_kv_quant.sh --dry-run  # show command only

set -e

ATLAS_HOME="/home/claude"
LOG_DIR="/tmp"
DRY_RUN=false
TARGET="spock"

for arg in "$@"; do
    case $arg in
        --kirk) TARGET="kirk" ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

if [ "$TARGET" = "spock" ]; then
    SESSION="qwen"  # current session name for the LLM on GPU 0
    GPU=0
    PORT=8080
    # Detect current model from running process
    MODEL=$(ps -eo args | grep "llama-server.*--port 8080" | grep -v grep | head -1 | sed -n 's/.*--model \([^ ]*\).*/\1/p' || true)
    if [ -z "$MODEL" ]; then
        echo "No llama-server running on port 8080. Using default."
        MODEL="$ATLAS_HOME/models/Qwen2.5-72B-Instruct-Q5_K_M/qwen2.5-72b-instruct-q5_k_m-00001-of-00014.gguf"
    fi
    THREADS=24
else
    SESSION="kirk"
    GPU=1
    PORT=8082
    MODEL="$ATLAS_HOME/models/Qwen3-32B-Q5_K_M/Qwen3-32B-Q5_K_M.gguf"
    THREADS=12
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU $ATLAS_HOME/llama.cpp/build/bin/llama-server \
    --model $MODEL \
    --host 0.0.0.0 --port $PORT --ctx-size 8192 --threads $THREADS \
    --cache-type-k q8_0 --cache-type-v q4_0 \
    --path $ATLAS_HOME/atlas-chat \
    2>&1 | tee $LOG_DIR/${TARGET}_kv_quant.log"

echo "=== KV Cache Quantization Restart ==="
echo "Target:  $TARGET ($SESSION session)"
echo "GPU:     $GPU"
echo "Port:    $PORT"
echo "Model:   $(basename $MODEL)"
echo "KV keys: q8_0 (8-bit, high precision for attention)"
echo "KV vals: q4_0 (4-bit, saves ~1.6 GB VRAM)"
echo ""
echo "Command:"
echo "  $CMD"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would execute the above. Run without --dry-run to apply."
    exit 0
fi

read -p "This will restart the LLM server on port $PORT. Continue? [y/N] " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

# Kill existing
echo "Stopping existing server on port $PORT..."
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 3

# Recreate tmux session
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$CMD"
echo "Started $TARGET with KV cache quantization in tmux session '$SESSION'"
echo ""
echo "Monitor: tmux attach -t $SESSION"
echo "Logs:    tail -f $LOG_DIR/${TARGET}_kv_quant.log"
