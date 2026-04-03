#!/bin/bash
# Atlas AI — Full System Startup Script
# Starts all AGI-HPC subsystems in the correct order
#
# Usage: bash scripts/start_atlas.sh
#        bash scripts/start_atlas.sh --stop
#        bash scripts/start_atlas.sh --health

set -e

ATLAS_HOME="/home/claude"
AGI_HOME="$ATLAS_HOME/agi-hpc"
VENV="$ATLAS_HOME/env/bin"
LOG_DIR="/tmp"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[Atlas]${NC} $1"; }
warn() { echo -e "${YELLOW}[Atlas]${NC} $1"; }
err() { echo -e "${RED}[Atlas]${NC} $1"; }
info() { echo -e "${CYAN}[Atlas]${NC} $1"; }

# ─── Health check mode ────────────────────────────────
if [ "$1" = "--health" ]; then
    info "Atlas AI — Health Check"
    info "============================================="

    # Load environment
    if [ -f "$ATLAS_HOME/.env" ]; then
        source "$ATLAS_HOME/.env"
    fi

    ALL_OK=true

    # Check tmux sessions
    SESSIONS=$(tmux ls 2>/dev/null | wc -l)
    info "  tmux sessions: $SESSIONS active"

    # NATS
    if curl -s http://localhost:8222/healthz >/dev/null 2>&1; then
        log "  NATS:          healthy (4222)"
    else
        err "  NATS:          DOWN"
        ALL_OK=false
    fi

    # Spock (Gemma 4)
    if curl -s http://localhost:8080/health 2>/dev/null | grep -q ok; then
        log "  Spock (LH):    healthy (8080)"
    else
        err "  Spock (LH):    DOWN"
        ALL_OK=false
    fi

    # Kirk (Qwen 3)
    if curl -s http://localhost:8082/health 2>/dev/null | grep -q ok; then
        log "  Kirk (RH):     healthy (8082)"
    else
        err "  Kirk (RH):     DOWN"
        ALL_OK=false
    fi

    # RAG Server (Flask app; check if port responds)
    if curl -s --max-time 3 http://localhost:8081/ >/dev/null 2>&1; then
        log "  RAG Server:    healthy (8081)"
    else
        err "  RAG Server:    DOWN"
        ALL_OK=false
    fi

    # PostgreSQL
    if systemctl is-active --quiet postgresql; then
        log "  PostgreSQL:    healthy (5432)"
    else
        err "  PostgreSQL:    DOWN"
        ALL_OK=false
    fi

    # Caddy
    if tmux has-session -t caddy 2>/dev/null; then
        log "  Caddy:         healthy (443)"
    else
        warn "  Caddy:         not running"
    fi

    # Memory Service
    if tmux has-session -t memory 2>/dev/null; then
        log "  Memory:        running"
    else
        warn "  Memory:        not running"
    fi

    # Safety Gateway
    if tmux has-session -t safety 2>/dev/null; then
        log "  Safety:        running"
    else
        warn "  Safety:        not running"
    fi

    # DHT Service
    if tmux has-session -t dht 2>/dev/null; then
        log "  DHT Registry:  running"
    else
        warn "  DHT Registry:  not running"
    fi

    # GPUs
    info ""
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | while read line; do
        info "  GPU: $line"
    done

    # DHT registry state (if available)
    DHT_STATUS=$(psql -d atlas -t -c "SELECT name, port, status FROM service_registry ORDER BY name" 2>/dev/null)
    if [ -n "$DHT_STATUS" ]; then
        info ""
        info "  DHT Service Registry:"
        echo "$DHT_STATUS" | while read line; do
            if [ -n "$line" ]; then
                info "    $line"
            fi
        done
    fi

    # Refresh DHT heartbeats for running services
    if tmux has-session -t dht 2>/dev/null; then
        PYTHONPATH=$AGI_HOME/src $VENV/python3 -c "
import asyncio, sys
sys.path.insert(0, '$AGI_HOME/src')

async def refresh():
    try:
        from agi.meta.dht.registry import ServiceRegistry
        registry = ServiceRegistry(dsn='dbname=atlas user=claude')
        await registry.init_db()
        services = {
            'nats': 4222, 'spock': 8080, 'kirk': 8082, 'rag': 8081,
            'memory': 50300, 'safety': 50055, 'caddy': 443, 'dht': 50080,
        }
        for name, port in services.items():
            await registry.heartbeat(name)
        await registry.close()
    except Exception:
        pass

asyncio.run(refresh())
" 2>/dev/null
        info "  DHT heartbeats refreshed"
    fi

    info ""
    if $ALL_OK; then
        log "All core services healthy."
    else
        err "Some core services are DOWN."
    fi
    exit 0
fi


# ─── Maintenance mode ──────────────────────────────────
if [ "$1" = "--maintenance" ] || [ "$1" = "--maint" ]; then
    log "Atlas AI — Entering Maintenance Mode"
    
    # Save original index.html
    [ ! -f "$ATLAS_HOME/atlas-chat/index.html.bak" ] && \
        cp "$ATLAS_HOME/atlas-chat/index.html" "$ATLAS_HOME/atlas-chat/index.html.bak"
    
    # Write maintenance page
    cat > "$ATLAS_HOME/atlas-chat/index.html" << 'MHTML'
<!DOCTYPE html><html><head><title>Atlas AI - Maintenance</title><meta name="viewport" content="width=device-width,initial-scale=1.0"><style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:system-ui;background:#0a0e17;color:#e0e6f0;height:100vh;display:flex;align-items:center;justify-content:center}.c{background:#131a2b;border:1px solid #2a3555;border-radius:16px;padding:40px;text-align:center;max-width:500px;width:90%}h1{font-size:28px;background:linear-gradient(135deg,#4a9eff,#7cc4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}p{color:#7a8ba8;margin:12px 0;font-size:14px}.s{display:inline-block;padding:6px 16px;background:rgba(245,158,11,0.15);border:1px solid #f59e0b;border-radius:20px;color:#f59e0b;font-size:13px;margin-top:16px}a{color:#4a9eff}</style></head><body><div class="c"><div style="font-size:48px;margin-bottom:16px">&#9881;</div><h1>Atlas AI</h1><p>Currently offline for scheduled maintenance.</p><div class="s">&#9202; Back soon</div><p style="margin-top:20px;font-size:12px"><a href="/schematic.html">Dashboard</a> | <a href="/events.html">Events</a></p></div></body></html>
MHTML
    
    # Kill LLM servers
    for s in spock kirk; do tmux kill-session -t $s 2>/dev/null && log "  Stopped $s"; done
    
    # Keep RAG for dashboard
    tmux has-session -t rag 2>/dev/null || \
        tmux new-session -d -s rag "CUDA_VISIBLE_DEVICES= $VENV/python3 $ATLAS_HOME/atlas-rag-server.py 2>&1 | tee $LOG_DIR/rag_server.log"
    
    log "  GPUs freed. Dashboard still accessible."
    log "  Exit maintenance: bash scripts/start_atlas.sh"
    exit 0
fi

# ─── Stop mode ──────────────────────────────────────────
if [ "$1" = "--stop" ]; then
    log "Stopping all Atlas services..."
    for session in nats spock kirk rag oauth2 caddy safety memory indexer dht; do
        tmux kill-session -t $session 2>/dev/null && log "  Stopped $session" || true
    done
    log "All services stopped."
    exit 0
fi

# Restore from maintenance if needed
[ -f "$ATLAS_HOME/atlas-chat/index.html.bak" ] && mv "$ATLAS_HOME/atlas-chat/index.html.bak" "$ATLAS_HOME/atlas-chat/index.html"

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

# DHT Service Registry (after NATS, before other services)
if [ -f "$AGI_HOME/src/agi/meta/dht/nats_service.py" ]; then
    if ! tmux has-session -t dht 2>/dev/null; then
        tmux new-session -d -s dht \
            "CUDA_VISIBLE_DEVICES= PYTHONPATH=$AGI_HOME/src $VENV/python3 -m agi.meta.dht.nats_service \
            --config $AGI_HOME/configs/dht_config.yaml \
            2>&1 | tee $LOG_DIR/dht_service.log"
        sleep 2
        log "  DHT Registry: started (service discovery)"
    else
        log "  DHT Registry: already running"
    fi
else
    warn "  DHT Registry: not yet implemented"
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
        "CUDA_VISIBLE_DEVICES= PYTHONPATH=$AGI_HOME/src $VENV/python3 -m agi.memory.nats_service \
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
            "CUDA_VISIBLE_DEVICES= PYTHONPATH=$AGI_HOME/src $VENV/python3 -m agi.safety.nats_service \
            2>&1 | tee $LOG_DIR/safety_service.log"
        sleep 2
        log "  Safety Gateway: started (DEME pipeline)"
    else
        log "  Safety Gateway: already running"
    fi
else
    warn "  Safety Gateway: not yet implemented"
fi

# ─── Register services with DHT ──────────────────────
if tmux has-session -t dht 2>/dev/null; then
    log ""
    log "Registering services with DHT..."
    # Give DHT a moment to be fully ready
    sleep 1

    # Register known services via direct DB insert (faster than NATS for startup)
    PYTHONPATH=$AGI_HOME/src $VENV/python3 -c "
import asyncio
import sys
sys.path.insert(0, '$AGI_HOME/src')

async def register_all():
    try:
        from agi.meta.dht.registry import ServiceRegistry
        registry = ServiceRegistry(dsn='dbname=atlas user=claude')  # parsed to kwargs internally
        await registry.init_db()

        services = {
            'nats':   (4222,  {'phase': 0, 'layer': 'infrastructure'}),
            'spock':  (8080,  {'phase': 1, 'layer': 'llm', 'model': 'gemma-4-31b'}),
            'kirk':   (8082,  {'phase': 1, 'layer': 'llm', 'model': 'qwen3-32b'}),
            'rag':    (8081,  {'phase': 2, 'layer': 'cognitive'}),
            'memory': (50300, {'phase': 2, 'layer': 'cognitive'}),
            'safety': (50055, {'phase': 3, 'layer': 'cognitive'}),
            'caddy':  (443,   {'phase': 0, 'layer': 'infrastructure'}),
            'dht':    (50080, {'phase': 6, 'layer': 'infrastructure'}),
        }

        for name, (port, meta) in services.items():
            await registry.register(name, port, meta)

        all_svc = await registry.list_all()
        print(f'  Registered {len(all_svc)} services with DHT')
        await registry.close()
    except Exception as e:
        print(f'  DHT registration warning: {e}')

asyncio.run(register_all())
" 2>/dev/null || warn "  DHT startup registration skipped (asyncpg not available)"
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
log "  DHT:         Service registry + config store"
log ""
log "  tmux sessions: $(tmux ls 2>/dev/null | wc -l) active"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | while read line; do
    log "  GPU: $line"
done
log ""
log "  Health:  bash scripts/start_atlas.sh --health"
log "  Stop:    bash scripts/start_atlas.sh --stop"
