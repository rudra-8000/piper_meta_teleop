"""Quest "headless" mode: make the headset believe it is being worn so a VR session survives you taking it off.

Meta's documented test broadcast ``com.oculus.vrpowermanager.prox_close`` simulates the proximity sensor seeing a face;
the community-documented ``automation_disable`` restores normal behaviour.  Both are verified here through the
headset's own property bundle (``set_proximity_close``), because a broadcast that was "completed" proves nothing.

Findings from a real Quest 3: the override works, BUT if the headset cannot localise itself (face-down, blocked lenses,
dark room) the system puts a "Finding position in room" dialog in front of every app, which blurs/ends the VR session.
"""
from __future__ import annotations

import re
import subprocess
import time
from typing import Callable

GUARDIAN_PKG = "com.oculus.guardian"
TRACKING_HINT = ("the headset cannot find its position in the room (its dialog says 'Finding position in room'). Stand it "
                 "UPRIGHT on a stand or tripod with the lenses facing the room - not face-down, lenses unobstructed, room "
                 "lit - so its cameras can see the room and your controllers. The dialog goes away by itself once it can "
                 "localise.")


def _subprocess_run(args, timeout):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)


class HeadlessManager:
    def __init__(self, run: Callable | None = None, settle: float = 0.8):
        self._run = run or _subprocess_run
        self.settle = settle              # seconds to let the headset react before verifying
        self.sent = False                 # we turned the override on (so we must turn it off again)

    # ------------------------------------------------------------------ adb
    def _adb(self, *args, timeout: float = 8.0):
        try:
            return self._run(["adb", *args], timeout)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            return 127, f"adb unavailable: {e}"

    def available(self) -> bool:
        rc, out = self._adb("devices")
        return rc == 0 and any(l.split()[1:2] == ["device"] for l in out.splitlines()[1:])

    def properties(self) -> dict | None:
        rc, out = self._adb("shell", "content", "call", "--uri", "content://com.oculus.rc", "--method", "GET_PROPERTY")
        m = re.search(r"Bundle\[\{(.*?)\}\]", out)
        if rc != 0 or not m:
            return None
        return {k.strip(): v.strip() for k, _, v in (kv.partition("=") for kv in m.group(1).split(","))}

    def foreground(self) -> str:
        rc, out = self._adb("shell", "dumpsys", "activity", "activities")
        m = re.search(r"topResumedActivity=ActivityRecord\{[^}]*? u0 ([\w./$]+)", out)
        return m.group(1) if m else ""

    def dialog_texts(self) -> list[str]:
        self._adb("shell", "uiautomator", "dump", "/sdcard/ui.xml")
        rc, out = self._adb("shell", "cat", "/sdcard/ui.xml")
        self._adb("shell", "rm", "-f", "/sdcard/ui.xml")
        return re.findall(r'text="([^"]+)"', out)

    # ------------------------------------------------------------- actions
    def on(self) -> dict:
        rc, out = self._adb("shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.prox_close")
        if rc != 0:
            return {"ok": False, "override": None, "foreground": "", "note": f"could not talk to the headset over adb ({out.strip()[:80]})"}
        self.sent = True
        time.sleep(self.settle)
        props = self.properties()
        override = None if props is None else props.get("set_proximity_close") == "true"
        fg = self.foreground()
        note = ""
        if override is False:
            note = "the headset did not accept the proximity override (set_proximity_close is false)."
        if GUARDIAN_PKG in fg:
            texts = " | ".join(self.dialog_texts())
            note = (TRACKING_HINT if "Finding position" in texts
                    else f"a Guardian dialog is covering the app ({texts[:120] or 'unknown text'}).")
        return {"ok": bool(override), "override": override, "foreground": fg, "note": note}

    def off(self) -> dict:
        rc, out = self._adb("shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.automation_disable")
        time.sleep(min(self.settle, 1.0))
        props = self.properties()
        restored = props is not None and props.get("set_proximity_close") == "false"
        if restored:
            self.sent = False
        return {"ok": restored, "note": "" if restored else
                "could not confirm the proximity override was removed - run: scripts/quest_adb.sh headless-off"}

    def off_if_active(self) -> dict | None:
        return self.off() if self.sent else None

    def status(self) -> dict:
        props = self.properties()
        return {"properties": props, "foreground": self.foreground()}
