#!/usr/bin/env bash
# Name + activate the AgileX USB-CAN adapters (1 Mbit/s).  Both arms need their OWN adapter.
#
#   scripts/can_setup.sh list                 show adapters and which USB port each is plugged into
#   scripts/can_setup.sh up NAME [USB_PORT]   rename/activate one adapter (USB_PORT like 1-2:1.0)
#   scripts/can_setup.sh all                  activate every "NAME USB_PORT" line in config/can_ports.conf
#   scripts/can_setup.sh persist              udev rule: fixed names + auto bring-up on (re)plug (needs sudo)
#   scripts/can_setup.sh down NAME
#
# Why USB ports: the two adapters are identical, so the only stable way to tell them apart is the
# physical USB port.  Plug each adapter into the SAME port every time.
set -euo pipefail
cd "$(dirname "$0")/.."
CONF=config/can_ports.conf
BITRATE=1000000

can_ifaces() { ip -br link show type can 2>/dev/null | awk '{print $1}'; }
bus_of()     { ethtool -i "$1" 2>/dev/null | awk '/bus-info/ {print $2}'; }

need() { command -v "$1" >/dev/null || { echo "missing '$1' - run scripts/install.sh (sudo apt install can-utils ethtool iproute2)"; exit 1; }; }
need ip; need ethtool

cmd_list() {
  local n=0
  for i in $(can_ifaces); do
    n=$((n+1)); printf '%-12s usb-port %-10s %s\n' "$i" "$(bus_of "$i")" "$(ip -br link show "$i" | awk '{print $2}')"
  done
  if [ "$n" -eq 0 ]; then
    echo "no CAN interfaces found."
    echo " - is the adapter plugged in?   (lsusb should show a 1d50:606f / 'candleLight'-style device)"
    echo " - is the driver loaded?        sudo modprobe gs_usb"
  else
    echo
    echo "Tip: unplug all adapters, plug in ONE, run 'list', write down its usb-port, repeat for the other."
  fi
}

cmd_up() {
  local name=$1 port=${2:-} iface=""
  if [ -n "$port" ]; then
    for i in $(can_ifaces); do [ "$(bus_of "$i")" = "$port" ] && iface=$i; done
    [ -n "$iface" ] || { echo "no CAN adapter on usb-port $port (see: $0 list)"; exit 1; }
  else
    [ "$(can_ifaces | wc -l)" -eq 1 ] || { echo "several adapters present: give a USB_PORT (see: $0 list)"; exit 1; }
    iface=$(can_ifaces)
  fi
  echo "-> $iface (usb $(bus_of "$iface")) becomes $name @ ${BITRATE} bit/s"
  sudo ip link set "$iface" down
  [ "$iface" = "$name" ] || sudo ip link set "$iface" name "$name"
  sudo ip link set "$name" type can bitrate "$BITRATE"
  sudo ip link set "$name" up
  ip -br link show "$name"
}

cmd_all() {
  [ -f "$CONF" ] || { echo "$CONF missing - copy config/can_ports.conf.example and fill in your USB ports"; exit 1; }
  while read -r name port _; do
    case "${name:-}" in ''|\#*) continue;; esac
    # already configured under this name? just make sure it is up
    if ip link show "$name" >/dev/null 2>&1; then
      sudo ip link set "$name" down; sudo ip link set "$name" type can bitrate "$BITRATE"; sudo ip link set "$name" up
      ip -br link show "$name"
    else
      cmd_up "$name" "$port"
    fi
  done < "$CONF"
}

cmd_persist() {
  [ -f "$CONF" ] || { echo "$CONF missing"; exit 1; }
  local rules=/etc/udev/rules.d/90-piper-can.rules tmp; tmp=$(mktemp)
  while read -r name port _; do
    case "${name:-}" in ''|\#*) continue;; esac
    # name it, and bring it up at 1 Mbit/s every time the adapter appears (e.g. after the arm is power-cycled)
    echo "SUBSYSTEM==\"net\", ACTION==\"add\", KERNELS==\"$port\", NAME=\"$name\", RUN+=\"$(command -v ip) link set $name type can bitrate $BITRATE\", RUN+=\"$(command -v ip) link set $name up\"" >> "$tmp"
  done < "$CONF"
  sudo install -m 644 "$tmp" "$rules"; rm -f "$tmp"
  sudo udevadm control --reload-rules
  echo "wrote $rules (names persist and links come up automatically whenever an adapter appears)"
  cat "$rules"
}

case "${1:-}" in
  list)    cmd_list ;;
  up)      [ $# -ge 2 ] || { sed -n 2,12p "$0"; exit 1; }; cmd_up "$2" "${3:-}" ;;
  all)     cmd_all ;;
  persist) cmd_persist ;;
  down)    sudo ip link set "$2" down ;;
  *)       sed -n 2,12p "$0"; exit 1 ;;
esac
