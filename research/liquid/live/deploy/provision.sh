#!/usr/bin/env bash
# Provision or update the C3 server (Debian 13, e2-micro). Idempotent; run as root:
#   sudo bash provision.sh /tmp/c3.bundle
# The bundle is `git bundle create c3.bundle research/live-r15` from a workstation.
# The keys go in /home/c3/QuantCode/.env by hand (ALPACA_*_C3 only), mode 600.
set -euo pipefail
bundle="${1:?usage: provision.sh <path to c3.bundle>}"
home=/home/c3
repo="$home/QuantCode"

timedatectl set-timezone America/New_York
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git sqlite3 python3 python3-venv ca-certificates >/dev/null

# 1 GB of swap: the e2-micro has 1 GB of memory.
if ! swapon --show=NAME --noheadings | grep -qx /swapfile; then
    [ -f /swapfile ] || { fallocate -l 1G /swapfile; chmod 600 /swapfile; mkswap /swapfile >/dev/null; }
    swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >>/etc/fstab
fi

id c3 >/dev/null 2>&1 || useradd --create-home --shell /bin/bash c3
install -o c3 -g c3 -m 644 "$bundle" "$home/c3.bundle"
if [ ! -d "$repo/.git" ]; then
    sudo -u c3 git clone -q -b research/live-r15 "$home/c3.bundle" "$repo"
else
    sudo -u c3 git -C "$repo" fetch -q "$home/c3.bundle" research/live-r15
    sudo -u c3 git -C "$repo" merge -q --ff-only FETCH_HEAD
fi

if [ ! -x "$repo/.venv/bin/python" ]; then sudo -u c3 python3 -m venv "$repo/.venv"; fi
sudo -u c3 "$repo/.venv/bin/pip" install -q --disable-pip-version-check \
    -r "$repo/research/liquid/live/deploy/requirements-c3.txt"

install -m 644 "$repo/research/liquid/live/deploy/c3.service" /etc/systemd/system/c3.service
install -m 644 "$repo/research/liquid/live/deploy/c3.timer" /etc/systemd/system/c3.timer
systemctl daemon-reload
systemctl enable --now c3.timer >/dev/null
echo "provisioned: $(sudo -u c3 git -C "$repo" log --oneline -1)"
systemctl list-timers c3.timer --no-pager
