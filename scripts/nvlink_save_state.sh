#!/bin/bash
# Save current ACS and NVLink state for rollback
set -e

echo "=== Saving NVLink pre-hack state ==="

ACS1=$(sudo setpci -s 00:02.0 ECAP_ACS+6.w 2>/dev/null || echo "unknown")
ACS2=$(sudo setpci -s 00:03.0 ECAP_ACS+6.w 2>/dev/null || echo "unknown")

cat > /tmp/nvlink_rollback.sh << ROLLBACK
#!/bin/bash
echo "Rolling back NVLink hack..."
sudo setpci -s 00:02.0 ECAP_ACS+6.w=${ACS1}
sudo setpci -s 00:03.0 ECAP_ACS+6.w=${ACS2}
sudo rm -f /etc/modprobe.d/nvlink-hack.conf
sudo systemctl stop nvidia-fabricmanager 2>/dev/null
sudo systemctl disable nvidia-fabricmanager 2>/dev/null
echo "Rollback complete. Reboot to fully restore."
ROLLBACK

chmod +x /tmp/nvlink_rollback.sh

echo "ACS saved: 00:02.0=$ACS1, 00:03.0=$ACS2"
echo "Rollback: bash /tmp/nvlink_rollback.sh"
nvidia-smi nvlink --status 2>&1 | head -5
echo "=== State saved ==="
