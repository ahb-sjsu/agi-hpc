#!/bin/bash
# Install Atlas systemd services.
#
# Replaces fragile tmux-based process management with proper systemd
# services that auto-restart on failure and survive reboots.
#
# Naming: Freudian psyche (Superego, Id, Ego), not Star Trek.
#
# Usage:
#   sudo bash deploy/systemd/install-services.sh
#   sudo bash deploy/systemd/install-services.sh --uninstall

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYSTEMD_DIR="/etc/systemd/system"

SERVICES=(
    atlas-nats
    atlas-superego
    atlas-id
    atlas-ego
    atlas-rag-server
    atlas-telemetry
    atlas-caddy
    atlas-oauth2-proxy
    atlas-watchdog
    atlas-primer
    atlas-scientist
    atlas-dreaming-schedule
)

# Old names to clean up
OLD_SERVICES=(
    atlas-llm-spock
    atlas-llm-kirk
    atlas-llm-dm
)

TIMERS=(
    atlas-training
    atlas-backup
)

TARGETS=(
    atlas
)

if [ "$1" = "--uninstall" ]; then
    echo "Stopping and removing Atlas services..."
    for svc in "${SERVICES[@]}" "${OLD_SERVICES[@]}"; do
        sudo systemctl stop "$svc" 2>/dev/null || true
        sudo systemctl disable "$svc" 2>/dev/null || true
        sudo rm -f "$SYSTEMD_DIR/$svc.service"
    done
    for timer in "${TIMERS[@]}"; do
        sudo systemctl stop "$timer.timer" 2>/dev/null || true
        sudo systemctl disable "$timer.timer" 2>/dev/null || true
        sudo rm -f "$SYSTEMD_DIR/$timer.service" "$SYSTEMD_DIR/$timer.timer"
    done
    for tgt in "${TARGETS[@]}"; do
        sudo systemctl disable "$tgt.target" 2>/dev/null || true
        sudo rm -f "$SYSTEMD_DIR/$tgt.target"
    done
    sudo systemctl daemon-reload
    echo "Done. All Atlas services removed."
    exit 0
fi

echo "============================================"
echo "Installing Atlas AI systemd services"
echo "Freudian Psyche: Superego + Id + Ego"
echo "============================================"

# Ensure log directory exists
mkdir -p /tmp/atlas

# Ensure COOKIE_SECRET exists in .env
if ! grep -q COOKIE_SECRET /home/claude/.env 2>/dev/null; then
    echo "Generating COOKIE_SECRET..."
    COOKIE=$(python3 -c "import os,base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode()[:32])")
    echo "COOKIE_SECRET=$COOKIE" >> /home/claude/.env
fi

# Remove old Spock/Kirk/DM service files
echo ""
echo "Cleaning up old service names..."
for old in "${OLD_SERVICES[@]}"; do
    if [ -f "$SYSTEMD_DIR/$old.service" ]; then
        sudo systemctl stop "$old" 2>/dev/null || true
        sudo systemctl disable "$old" 2>/dev/null || true
        sudo rm -f "$SYSTEMD_DIR/$old.service"
        echo "  Removed old $old.service"
    fi
done

# Stop tmux-based services (they'll be replaced by systemd)
echo ""
echo "Stopping tmux-based services..."
for session in spock kirk ego rag caddy oauth2 telemetry nats safety memory dht; do
    tmux kill-session -t "$session" 2>/dev/null && echo "  Stopped tmux:$session" || true
done
sleep 3

# Kill leftover processes
pkill -f "oauth2-proxy" 2>/dev/null || true
pkill -f "atlas-rag-server" 2>/dev/null || true
pkill -f "telemetry_server" 2>/dev/null || true
sleep 2

# Install service files
echo ""
echo "Installing service files..."
for svc in "${SERVICES[@]}"; do
    src="$SCRIPT_DIR/$svc.service"
    if [ -f "$src" ]; then
        sudo cp "$src" "$SYSTEMD_DIR/"
        echo "  Installed $svc.service"
    else
        echo "  WARNING: $src not found, skipping"
    fi
done

# Install target files
echo ""
echo "Installing target files..."
for tgt in "${TARGETS[@]}"; do
    src="$SCRIPT_DIR/$tgt.target"
    if [ -f "$src" ]; then
        sudo cp "$src" "$SYSTEMD_DIR/"
        echo "  Installed $tgt.target"
    fi
done

# Install timer files
echo ""
echo "Installing timer files..."
for timer in "${TIMERS[@]}"; do
    for ext in service timer; do
        src="$SCRIPT_DIR/$timer.$ext"
        if [ -f "$src" ]; then
            sudo cp "$src" "$SYSTEMD_DIR/"
            echo "  Installed $timer.$ext"
        fi
    done
done

# Reload systemd
sudo systemctl daemon-reload

# Enable and start services (in dependency order)
echo ""
echo "Starting services..."
START_ORDER=(
    atlas-nats
    atlas-telemetry
    atlas-superego
    atlas-id
    atlas-ego
    atlas-rag-server
    atlas-oauth2-proxy
    atlas-caddy
    atlas-watchdog
    atlas-primer
    atlas-scientist
    atlas-dreaming-schedule
)
for svc in "${START_ORDER[@]}"; do
    if [ -f "$SYSTEMD_DIR/$svc.service" ]; then
        sudo systemctl enable "$svc" 2>/dev/null
        sudo systemctl start "$svc" 2>/dev/null || true
        sleep 2
        STATUS=$(sudo systemctl is-active "$svc" 2>/dev/null || echo "failed")
        echo "  $svc: $STATUS"
    fi
done

# Enable target
echo ""
echo "Enabling atlas.target..."
sudo systemctl enable atlas.target 2>/dev/null || true

# Enable timers
echo ""
echo "Enabling timers..."
for timer in "${TIMERS[@]}"; do
    if [ -f "$SYSTEMD_DIR/$timer.timer" ]; then
        sudo systemctl enable "$timer.timer" 2>/dev/null
        sudo systemctl start "$timer.timer" 2>/dev/null || true
        echo "  $timer.timer: enabled"
    fi
done

# Summary
echo ""
echo "============================================"
echo "Atlas AI services installed"
echo "============================================"
echo ""
echo "Services:"
for svc in "${START_ORDER[@]}"; do
    STATUS=$(sudo systemctl is-active "$svc" 2>/dev/null || echo "inactive")
    printf "  %-25s %s\n" "$svc" "$STATUS"
done
echo ""
echo "Timers:"
for timer in "${TIMERS[@]}"; do
    STATUS=$(sudo systemctl is-active "$timer.timer" 2>/dev/null || echo "inactive")
    printf "  %-25s %s\n" "$timer.timer" "$STATUS"
done
echo ""
echo "Commands:"
echo "  sudo systemctl status atlas-*              # check all"
echo "  sudo systemctl restart atlas-superego       # restart one"
echo "  sudo systemctl start atlas.target           # start everything"
echo "  sudo journalctl -u atlas-superego -f        # follow logs"
echo "  sudo bash $SCRIPT_DIR/install-services.sh --uninstall"
echo ""
