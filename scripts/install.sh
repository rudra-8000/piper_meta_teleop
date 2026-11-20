#!/usr/bin/env bash
# One-shot setup for Ubuntu 22.04/24.04: system tools + Python venv.  Needs sudo for apt only.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> system packages (can-utils, ethtool, iproute2, adb, openssl, python venv)"
sudo apt-get update
sudo apt-get install -y can-utils ethtool iproute2 android-tools-adb openssl python3-venv git

echo "==> python venv in .venv"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "==> make sure the gs_usb CAN driver is available"
if ! modinfo gs_usb >/dev/null 2>&1; then
  echo "WARNING: kernel module gs_usb not found - the AgileX USB-CAN adapter will not appear as canX."
  echo "         On stock Ubuntu kernels it ships with linux-modules-extra:  sudo apt install linux-modules-extra-\$(uname -r)"
fi
sudo modprobe gs_usb || true

echo "==> self test (no hardware needed)"
.venv/bin/python -m pytest -q tests

cat <<MSG

Done.  Next:
  1. Read docs/Piper_Setup_Guide.pdf (unboxing -> power -> CAN).
  2. scripts/can_setup.sh list
  3. .venv/bin/python tools/hw_check.py can_left can_right
MSG
