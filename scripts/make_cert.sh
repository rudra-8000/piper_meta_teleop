#!/usr/bin/env bash
# Self-signed TLS certificate so the Quest browser can load the page over https:// on Wi-Fi.
# (Not needed for USB: http://localhost counts as a secure context for WebXR.)
#   scripts/make_cert.sh [extra-ip ...]     # PC's LAN addresses are added automatically
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p certs
san="DNS:localhost,IP:127.0.0.1"
for ip in $(hostname -I) "$@"; do san="$san,IP:$ip"; done
openssl req -x509 -newkey rsa:2048 -nodes -days 825 -keyout certs/key.pem -out certs/cert.pem \
  -subj "/CN=piper-teleop" -addext "subjectAltName=$san" 2>/dev/null
chmod 600 certs/key.pem
echo "created certs/cert.pem + certs/key.pem for: $san"
echo "On the Quest: open https://<PC-LAN-IP>:8788/ , tap Advanced -> Proceed (once)."
