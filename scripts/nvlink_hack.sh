#!/bin/bash
# NVLink hack for HP Z840 — try to enable NVLink across CPU sockets
#
# Three approaches, tried in order:
#   1. Disable ACS on PCIe root ports (live, instant, reversible)
#   2. Install nvidia-fabricmanager (service, reversible)
#   3. Set GrdmaPciTopoCheckOverride (module param, needs reboot)
#
# Rollback: bash /tmp/nvlink_rollback.sh
#
# WARNING: This is experimental. The NVLink bridge is physically
# installed but the Z840 BIOS doesn't negotiate it.

set -e

check_nvlink() {
    echo "--- NVLink status ---"
    nvidia-smi nvlink --status 2>&1 | head -6
    echo ""
}

echo "=========================================="
echo "NVLink Hack for HP Z840"
echo "=========================================="
echo ""

# Baseline
echo "BEFORE:"
check_nvlink

# ─── Attempt 1: Disable ACS ───────────────────────────────────────
echo "=== Attempt 1: Disable ACS on PCIe root ports ==="
echo "Setting ACS control to 0 on 00:02.0 and 00:03.0..."
sudo setpci -s 00:02.0 ECAP_ACS+6.w=0000
sudo setpci -s 00:03.0 ECAP_ACS+6.w=0000
echo "ACS disabled."
sleep 2
echo "AFTER ACS disable:"
check_nvlink

RESULT=$(nvidia-smi nvlink --status 2>&1 | grep -c "inActive")
if [ "$RESULT" = "0" ]; then
    echo "*** NVLink ACTIVE! ACS was the blocker. ***"
    exit 0
fi

# ─── Attempt 2: Install fabric-manager ────────────────────────────
echo "=== Attempt 2: Install nvidia-fabricmanager-570 ==="
if ! dpkg -l nvidia-fabricmanager-570 2>/dev/null | grep -q "^ii"; then
    echo "Installing..."
    sudo apt-get install -y nvidia-fabricmanager-570 2>&1 | tail -5
else
    echo "Already installed."
fi
echo "Starting fabric-manager..."
sudo systemctl start nvidia-fabricmanager 2>/dev/null || true
sleep 3
echo "AFTER fabric-manager:"
check_nvlink

RESULT=$(nvidia-smi nvlink --status 2>&1 | grep -c "inActive")
if [ "$RESULT" = "0" ]; then
    echo "*** NVLink ACTIVE! Fabric-manager did it. ***"
    sudo systemctl enable nvidia-fabricmanager
    exit 0
fi

# ─── Attempt 3: GrdmaPciTopoCheckOverride ────────────────────────
echo "=== Attempt 3: GrdmaPciTopoCheckOverride ==="
echo "Writing modprobe config (takes effect on next driver reload)..."
echo 'options nvidia NVreg_GrdmaPciTopoCheckOverride=1' | sudo tee /etc/modprobe.d/nvlink-hack.conf
echo ""
echo "This parameter requires reloading the nvidia module."
echo "To apply without reboot, you would need to:"
echo "  1. Stop all GPU processes (LLMs, etc)"
echo "  2. sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia"
echo "  3. sudo modprobe nvidia"
echo "  4. Restart GPU processes"
echo ""
echo "Or just reboot when convenient."
echo ""
echo "CURRENT STATUS (won't change until module reload):"
check_nvlink

echo "=========================================="
echo "Summary:"
echo "  ACS disabled: YES"
echo "  Fabric-manager: installed + started"
echo "  TopoCheckOverride: configured (needs module reload)"
echo "  Rollback: bash /tmp/nvlink_rollback.sh"
echo "=========================================="
