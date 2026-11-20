#!/usr/bin/env bash
# One-time: let your normal user talk to the Meta Quest over adb (fixes "no permissions" in `adb devices`).
# Meta/Oculus USB vendor id is 2833 (your Quest 3 shows up as 2833:5013 in lsusb).
set -euo pipefail
RULE=/etc/udev/rules.d/51-meta-quest-adb.rules
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0660", GROUP="plugdev", TAG+="uaccess"' | sudo tee "$RULE"
sudo udevadm control --reload-rules
sudo udevadm trigger
adb kill-server 2>/dev/null || true
adb start-server
echo
echo "Done.  Unplug and re-plug the Quest's USB cable, then:  scripts/quest_adb.sh check"
echo "(you must be in the plugdev group - check with: id)"
