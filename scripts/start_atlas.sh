#!/bin/bash
# Atlas AI — Full System Startup Script
# Starts all AGI-HPC subsystems in the correct order
#
# Usage: bash scripts/start_atlas.sh
#        bash scripts/start_atlas.sh --stop

set -e

ATLAS_HOME="/home/claude"
AGI_HOME="$ATLAS_HOME/agi-hpc"
VENV="$ATLAS_HOME/env/bin"
LOG_DIR="/tmp"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[Atlas]${NC} $1"; }
warn() { echo -e "${YELLOW}[Atlas]${NC} $1"; }
err() { echo -e "${RED}[Atlas]${NC} $1"; }

# ─── Stop mode ──────────────────────────────────────────
if [ "$1" = "--stop" ]; then
    log "Stopping all Atlas services..."
    for session in nats spock kirk rag oauth2 caddy safety memory indexer; do
        tmux kill-session -t $session 2>/dev/null && log "  Stopped $session" || true
    done
    log "All services stopped."
    exit 0
fi

# ─── Pre-flight checks ─────────────────────────────────
log "Atlas AI — Starting AGI-HPC Cognitive Architecture"
log "============================================="

# Check RAID mounts
if ! mountpoint -q /archive; then
    warn "  /archive not mounted, attempting mount..."
    sudo mount /archive 2>/dev/null || err "  Failed to mount /archive"
fi
if ! mountpoint -q /mnt/newhome; then
    warn "  /mnt/newhome not mounted, attempting mount..."
    sudo mount /mnt/newhome 2>/dev/null || err "  Failed to mount /mnt/newhome"
fi

# Check GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
log "  GPUs: $GPU_COUNT detected"

# Check PostgreSQL
if systemctl is-active --quiet postgresql; then
    log "  PostgreSQL: running"
else
    warn "  PostgreSQL: starting..."
    sudo systemctl start postgresql
fi

# ─── Layer 0: Infrastructure ───────────────────────────
log ""
log "Layer 0: Infrastructure"

# NATS
if ! tmux has-session -t nats 2>/dev/null; then
    tmux new-session -d -s nats \
        "$ATLAS_HOME/bin/nats-server --jetstream --store_dir $ATLAS_HOME/nats-data --addr 0.0.0.0 --port 4222 --http_port 8222 2>&1 | tee $LOG_DIR/nats.log"
    sleep 2
    log "  NATS: started (port 4222, monitoring 8222)"
else
    log "  NATS: already running"
fi

# Caddy (HTTPS reverse proxy)
if ! tmux has-session -t caddy 2>/dev/null; then
    tmux new-session -d -s caddy \
        "sudo caddy run --config $ATLAS_HOME/Caddyfile 2>&1 | tee $LOG_DIR/caddy.log"
    sleep 2
    log "  Caddy: started (HTTPS on 443)"
else
    log "  Caddy: already running"
fi

# OAuth2 Proxy
if ! tmux has-session -t oauth2 2>/dev/null; then
    # OAuth credentials from environment or .env file
    if [ -f "$ATLAS_HOME/.env" ]; then
        source "$ATLAS_HOME/.env"
    fi
    if [ -z "$GOOGLE_CLIENT_ID" ] || [ -z "$GOOGLE_CLIENT_SECRET" ]; then
        err "  Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in ~/.env"
        err "  Example: echo 'GOOGLE_CLIENT_ID=xxx' >> ~/.env"
    else
        COOKIE_SECRET=$(openssl rand -base64 24)
        tmux new-session -d -s oauth2 \
            "oauth2-proxy \
            --provider=google \
            --client-id=$GOOGLE_CLIENT_ID \
            --client-secret=$GOOGLE_CLIENT_SECRET \
            --cookie-secret=$COOKIE_SECRET \
            --cookie-secure=true \
            --upstream=http://localhost:8081 \
            --http-address=0.0.0.0:4180 \
            --redirect-url=https://atlas-sjsu.duckdns.org/oauth2/callback \
            --email-domain=* \
            --upstream-timeout=300s \
            --custom-templates-dir=$ATLAS_HOME/oauth2-templates \
            --banner=- --footer=- \
            2>&1 | tee $LOG_DIR/oauth2-proxy.log"
    fi
    sleep 2
    log "  OAuth2 Proxy: started (port 4180)"
else
    log "  OAuth2 Proxy: already running"
fi

# ─── Layer 1: LLM Engines ─────────────────────────────
log ""
log "Layer 1: LLM Engines (Dual Hemisphere)"

# Spock (Gemma 4 31B) on GPU 0
if ! tmux has-session -t spock 2>/dev/null; then
    tmux new-session -d -s spock \
        "CUDA_VISIBLE_DEVICES=0 $ATLAS_HOME/llama.cpp/build/bin/llama-server \
        --model $ATLAS_HOME/models/gemma-4-31B-it-Q5_K_M/gemma-4-31B-it-Q5_K_M.gguf \
        --host 0.0.0.0 --port 8080 --ctx-size 8192 --threads 12 \
        --path $ATLAS_HOME/atlas-chat \
        2>&1 | tee $LOG_DIR/spock.log"
    log "  Spock (Gemma 4 31B): starting on GPU 0:8080..."
else
    log "  Spock: already running"
fi

# Kirk (Qwen 3 32B) on GPU 1
if ! tmux has-session -t kirk 2>/dev/null; then
    tmux new-session -d -s kirk \
        "CUDA_VISIBLE_DEVICES=1 $ATLAS_HOME/llama.cpp/build/bin/llama-server \
        --model $ATLAS_HOME/models/Qwen3-32B-Q5_K_M/Qwen3-32B-Q5_K_M.gguf \
        --host 0.0.0.0 --port 8082 --ctx-size 8192 --threads 12 \
        2>&1 | tee $LOG_DIR/kirk.log"
    log "  Kirk (Qwen 3 32B): starting on GPU 1:8082..."
else
    log "  Kirk: already running"
fi

# Wait for models to load
log "  Waiting for models to load..."
for i in $(seq 1 30); do
    SPOCK_OK=$(curl -s http://localhost:8080/health 2>/dev/null | grep -c ok || true)
    KIRK_OK=$(curl -s http://localhost:8082/health 2>/dev/null | grep -c ok || true)
    if [ "$SPOCK_OK" = "1" ] && [ "$KIRK_OK" = "1" ]; then
        log "  Both hemispheres online!"
        break
    fi
    sleep 5
done

# ─── Layer 2: Cognitive Services ───────────────────────
log ""
log "Layer 2: Cognitive Services"

# RAG Server (integration + routing)
if ! tmux has-session -t rag 2>/dev/null; then
    tmux new-session -d -s rag \
        "CUDA_VISIBLE_DEVICES= $VENV/python3 $ATLAS_HOME/atlas-rag-server.py \
        2>&1 | tee $LOG_DIR/rag_server.log"
    sleep 5
    log "  RAG Server: started (port 8081, dual-hemisphere routing)"
else
    log "  RAG Server: already running"
fi

# Memory Service
if ! tmux has-session -t memory 2>/dev/null; then
    tmux new-session -d -s memory \
        "CUDA_VISIBLE_DEVICES= $VENV/python3 -m agi.memory.nats_service \
        2>&1 | tee $LOG_DIR/memory_service.log"
    sleep 2
    log "  Memory Service: started (episodic + procedural + semantic)"
else
    log "  Memory Service: already running"
fi

# Safety Gateway (if available)
if [ -f "$AGI_HOME/src/agi/safety/nats_service.py" ]; then
    if ! tmux has-session -t safety 2>/dev/null; then
        tmux new-session -d -s safety \
            "CUDA_VISIBLE_DEVICES= $VENV/python3 -m agi.safety.nats_service \
            2>&1 | tee $LOG_DIR/safety_service.log"
        sleep 2
        log "  Safety Gateway: started (DEME pipeline)"
    else
        log "  Safety Gateway: already running"
    fi
else
    warn "  Safety Gateway: not yet implemented"
fi

# ─── Summary ──────────────────────────────────────────
log ""
log "============================================="
log "Atlas AI is online!"
log ""
log "  Chat:        https://atlas-sjsu.duckdns.org"
log "  Schematic:   https://atlas-sjsu.duckdns.org/schematic.html"
log "  NATS Monitor: http://localhost:8222"
log ""
log "  Spock (LH):  Gemma 4 31B on GPU 0"
log "  Kirk  (RH):  Qwen 3 32B on GPU 1"
log "  Memory:      PostgreSQL + pgvector + PostGIS"
log "  RAG:         Hybrid search + HyDE over 27 repos"
log ""
log "  tmux sessions: $(tmux ls 2>/dev/null | wc -l) active"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | while read line; do
    log "  GPU: $line"
done
log ""
log "  Stop all: bash scripts/start_atlas.sh --stop"
