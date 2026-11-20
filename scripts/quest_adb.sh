#!/usr/bin/env bash
# Quest 3 helper over USB (adb).  Needs Developer Mode on the Quest and "Allow USB debugging" accepted in-headset.
#
#   scripts/quest_adb.sh check             is the headset visible / authorised?
#   scripts/quest_adb.sh usb [PORT]        adb reverse PORT (default: transport.port in config/teleop.yaml) + open the page
#                                          (also accepted: usb --port PORT)
#   scripts/quest_adb.sh headless-on       headset believes it is being worn even when it is on a stand (verified)
#   scripts/quest_adb.sh headless-off      back to normal proximity-sensor behaviour (verified)
#   scripts/quest_adb.sh headless-status   override on/off, what is in front, and any dialog text
#   (the teleop server does headless-on/off by itself for a headless session; these are for manual use)
#   scripts/quest_adb.sh wifi              switch adb to TCP/IP (USB-free control channel; data link is separate)
#
# headless-on uses Meta's documented test broadcast (com.oculus.vrpowermanager.prox_close).  The restore
# command (automation_disable) is only community-documented; prox_far is the documented opposite.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=$(awk '/^  port:/ {print $2; exit}' config/teleop.yaml 2>/dev/null)
PORT=${PORT:-8788}
shift_args=("${@:2}")
while [ ${#shift_args[@]} -gt 0 ]; do
  case "${shift_args[0]}" in
    --port) PORT=${shift_args[1]:?--port needs a number}; shift_args=("${shift_args[@]:2}") ;;
    [0-9]*) PORT=${shift_args[0]}; shift_args=("${shift_args[@]:1}") ;;
    *) echo "unknown argument: ${shift_args[0]}"; exit 1 ;;
  esac
done
command -v adb >/dev/null || { echo "adb missing: sudo apt install android-tools-adb"; exit 1; }

dev_ok() {
  local out n; out=$(adb devices)
  if grep -q "no permissions" <<<"$out"; then
    echo "$out"; echo
    echo "Linux is blocking USB access to the Quest (no udev rule).  Fix once with:  scripts/quest_udev.sh"
    echo "then unplug/replug the USB cable and run this again."
    exit 1
  fi
  if grep -q "unauthorized" <<<"$out"; then
    echo "$out"; echo "Put the headset on and accept 'Allow USB debugging' (tick always allow), then retry."; exit 1
  fi
  n=$(awk 'NR>1 && $2=="device"' <<<"$out" | wc -l)
  if [ "$n" -eq 0 ]; then
    adb devices
    echo "No authorised Quest found.  (1) Quest mobile app > Devices > Developer Mode ON  (2) USB cable in,"
    echo "(3) put the headset on and accept 'Allow USB debugging' (tick 'always allow')."
    exit 1
  fi
  [ "$n" -eq 1 ] || { echo "more than one adb device; unplug the others"; exit 1; }
}

case "${1:-}" in
  check)
    dev_ok; adb shell getprop ro.product.model; adb shell getprop ro.build.version.release
    echo "Quest is reachable."
    ;;
  usb)
    dev_ok
    adb reverse "tcp:$PORT" "tcp:$PORT"
    echo "adb reverse active: Quest localhost:$PORT -> this PC:$PORT"
    adb shell am start -a android.intent.action.VIEW -d "http://localhost:$PORT/" com.oculus.browser >/dev/null || true
    echo "Opened http://localhost:$PORT/ in Quest Browser.  If nothing opened, type that URL in the browser by hand."
    sleep 2
    if adb shell dumpsys power 2>/dev/null | grep -q "mWakefulness=Asleep"; then
      echo "NOTE: the headset is ASLEEP (it sleeps within seconds when it is not on your head), so the page cannot stream."
      echo "      Put it on, then press 'Enter VR & start streaming'.  Without wearing it: $0 headless-on  (see the guide)."
    fi
    ;;
  headless-status)
    dev_ok
    echo "properties : $(adb shell content call --uri content://com.oculus.rc --method GET_PROPERTY 2>&1 | head -1)"
    echo "in front   : $(adb shell dumpsys activity activities 2>/dev/null | grep -m1 topResumedActivity | sed 's/.* u0 //; s/ .*//')"
    adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1
    texts=$(adb shell cat /sdcard/ui.xml 2>/dev/null | grep -o 'text="[^"]\+"' | head -6 | tr '\n' ' ')
    adb shell rm -f /sdcard/ui.xml >/dev/null 2>&1
    [ -n "$texts" ] && echo "dialog text: $texts"
    ;;
  headless-on)
    dev_ok
    adb shell am broadcast -a com.oculus.vrpowermanager.prox_close >/dev/null
    sleep 2
    if adb shell content call --uri content://com.oculus.rc --method GET_PROPERTY 2>&1 | grep -q "set_proximity_close=true"; then
      echo "Proximity override ON: the headset now treats itself as worn."
    else
      echo "WARNING: the headset did not accept the override (set_proximity_close is not true)."; exit 1
    fi
    if adb shell dumpsys activity activities 2>/dev/null | grep -m1 topResumedActivity | grep -q "com.oculus.guardian"; then
      echo "NOTE: a Guardian dialog is in front.  If it says 'Finding position in room', the headset cannot see the room:"
      echo "      stand it UPRIGHT on a stand/tripod, lenses facing the room (not face-down), room lit.  It clears by itself."
    fi
    echo "Run '$0 headless-off' when finished (or stop the teleop server, which restores it)."
    ;;
  headless-off)
    dev_ok
    adb shell am broadcast -a com.oculus.vrpowermanager.automation_disable >/dev/null || true
    sleep 2
    if adb shell content call --uri content://com.oculus.rc --method GET_PROPERTY 2>&1 | grep -q "set_proximity_close=false"; then
      echo "Proximity override OFF: normal behaviour restored (the screen turns off when the headset is not worn)."
    else
      echo "Could not confirm the override was removed.  Try again, or: adb shell am broadcast -a com.oculus.vrpowermanager.prox_far"
      exit 1
    fi
    ;;
  wifi)
    dev_ok
    ip=$(adb shell ip -f inet addr show wlan0 | awk '/inet / {sub("/.*","",$2); print $2}')
    adb tcpip 5555; sleep 1; adb connect "$ip:5555"
    echo "adb over Wi-Fi at $ip:5555 (you can unplug USB; 'adb reverse' still works over this link)."
    ;;
  *) sed -n 2,16p "$0"; exit 1 ;;
esac
