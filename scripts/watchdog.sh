#!/bin/bash
# watchdog.sh — Atlas health monitor
#
# Periodically checks all Atlas services and logs status.
# Systemd handles actual restarts (Restart=always on each service);
# this script provides unified health reporting and alerting.
#
# Runs as atlas-watchdog.service, checks every 60 seconds.

set -uo pipefail

INTERVAL=60
LOG_PREFIX="[watchdog]"

# Services to monitor (name:port:endpoint[:ok_codes]); ok_codes defaults to 200.
#
# Superego (atlas-superego, Gemma 31B on GPU 0) is intentionally off; it is
# not checked. Id on 8082 is served by nrp_llm_proxy.py (NRP-hosted model),
# not atlas-id: the proxy answers only POST, so a 501 to GET means it is up.
SERVICES=(
    "Id:8082:/health:200,501"
    "Ego:8084:/health"
    "RAG:8081:/api/search-status"
    "NATS:8222:/varz"
    "Telemetry:8085:/api/health"
)

SYSTEMD_SERVICES=(
    "atlas-ego"
    "atlas-rag-server"
    "atlas-nats"
    "atlas-telemetry"
    "atlas-caddy"
    "atlas-oauth2-proxy"
)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_PREFIX $1"
}

check_http() {
    local name="$1" port="$2" endpoint="$3" ok_codes="${4:-200}"
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" \
        "http://localhost:${port}${endpoint}" \
        --connect-timeout 3 --max-time 5 2>/dev/null || echo "000")
    if [[ ",$ok_codes," == *",$status,"* ]]; then
        echo "online"
    else
        echo "offline($status)"
    fi
}

check_systemd() {
    local svc="$1"
    systemctl is-active "$svc" 2>/dev/null || echo "unknown"
}

log "=== Atlas Watchdog Started ==="
log "Monitoring ${#SERVICES[@]} HTTP endpoints, ${#SYSTEMD_SERVICES[@]} systemd services"
log "Check interval: ${INTERVAL}s"

while true; do
    ONLINE=0
    OFFLINE=0
    STATUS_LINE=""

    # HTTP health checks
    for entry in "${SERVICES[@]}"; do
        IFS=: read -r name port endpoint ok_codes <<< "$entry"
        result=$(check_http "$name" "$port" "$endpoint" "$ok_codes")
        if [ "$result" = "online" ]; then
            ONLINE=$((ONLINE + 1))
        else
            OFFLINE=$((OFFLINE + 1))
            log "WARNING: $name ($port) is $result"
        fi
        STATUS_LINE="$STATUS_LINE $name=$result"
    done

    # Systemd service checks
    SYSTEMD_DOWN=""
    for svc in "${SYSTEMD_SERVICES[@]}"; do
        state=$(check_systemd "$svc")
        if [ "$state" != "active" ]; then
            SYSTEMD_DOWN="$SYSTEMD_DOWN $svc($state)"
        fi
    done

    # Log summary
    TOTAL=$((ONLINE + OFFLINE))
    if [ "$OFFLINE" -eq 0 ] && [ -z "$SYSTEMD_DOWN" ]; then
        # All healthy — log less frequently (every 5th check)
        if [ $((SECONDS % (INTERVAL * 5))) -lt "$INTERVAL" ]; then
            log "OK: $ONLINE/$TOTAL services online |$STATUS_LINE"
        fi
    else
        log "DEGRADED: $ONLINE/$TOTAL online, $OFFLINE offline |$STATUS_LINE"
        if [ -n "$SYSTEMD_DOWN" ]; then
            log "Systemd down:$SYSTEMD_DOWN"
        fi
    fi

    # GPU thermal check
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | sort -rn | head -1)
    if [ -n "$GPU_TEMP" ] && [ "$GPU_TEMP" -gt 85 ]; then
        log "THERMAL WARNING: GPU at ${GPU_TEMP}°C (threshold: 85°C)"
    fi

    # CPU thermal check
    # Hottest package's current reading only (each line also carries high/crit).
    CPU_TEMP=$(sensors 2>/dev/null | grep -oP '^Package id \d+:\s+\+\K[0-9.]+' | sort -rn | head -1)
    CPU_TEMP=${CPU_TEMP:-0}
    CPU_INT=${CPU_TEMP%.*}
    if [ -n "$CPU_INT" ] && [ "$CPU_INT" -gt 90 ]; then
        log "THERMAL WARNING: CPU at ${CPU_TEMP}°C (threshold: 90°C)"
    fi

    sleep "$INTERVAL"
done
