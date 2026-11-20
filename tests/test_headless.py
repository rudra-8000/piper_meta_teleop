import json
import time

import pytest
from websockets.sync.client import connect

from piper_teleop.headless import HeadlessManager
from test_e2e import server  # noqa: F401  (shared fixture: real server + simulated arms)


class FakeAdb:
    """Pretends to be adb talking to a Quest."""

    def __init__(self, accept=True, guardian=None, devices=True):
        self.prox, self.accept, self.guardian, self.devices = False, accept, guardian, devices
        self.log = []

    def run(self, args, timeout):
        self.log.append(" ".join(args))
        a = args[1:]
        if a[:1] == ["devices"]:
            return 0, "List of devices attached\n" + ("XYZ\tdevice\n" if self.devices else "")
        if not self.devices:
            return 1, "error: no devices"
        if a[:5] == ["shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.prox_close"]:
            self.prox = self.accept
            return 0, "Broadcast completed: result=0"
        if a[:5] == ["shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.automation_disable"]:
            self.prox = False
            return 0, "Broadcast completed: result=0"
        if "GET_PROPERTY" in a:
            return 0, ("Result: Bundle[{disable_guardian=false, set_proximity_close=%s, disable_dialogs=false, "
                       "disable_autosleep=false}]" % str(self.prox).lower())
        if a[:3] == ["shell", "dumpsys", "activity"]:
            front = ("com.oculus.guardian/com.oculus.vrguardianservice.guardiandialog.GuardianDialogActivity"
                     if self.guardian else "com.oculus.browser/.BrowserActivity")
            return 0, f"  topResumedActivity=ActivityRecord{{77c5ef7 u0 {front} t1267}}\n"
        if a[:2] == ["shell", "cat"]:
            return 0, "".join(f'<node text="{t}"/>' for t in (self.guardian or []))
        return 0, ""


def mgr(**kw):
    f = FakeAdb(**kw)
    return HeadlessManager(run=f.run, settle=0.0), f


def test_on_verifies_the_override_through_the_headsets_own_property():
    m, f = mgr()
    r = m.on()
    assert r["ok"] and r["override"] is True and r["note"] == "" and m.sent
    m2, f2 = mgr(accept=False)
    r2 = m2.on()
    assert not r2["ok"] and "did not accept" in r2["note"]


def test_tracking_lost_dialog_is_recognised_and_explained():
    m, f = mgr(guardian=["Finding position in room", "Continue without tracking", "Turn on travel mode"])
    r = m.on()
    assert r["ok"] and "Finding position in room" in r["note"] and "UPRIGHT" in r["note"] and "lit" in r["note"]


def test_other_guardian_dialogs_are_reported_with_their_text():
    m, _ = mgr(guardian=["Boundary not set"])
    assert "Boundary not set" in m.on()["note"]


def test_no_headset_or_no_adb_is_reported_not_raised():
    m, _ = mgr(devices=False)
    r = m.on()
    assert not r["ok"] and "adb" in r["note"] and not m.sent
    m3 = HeadlessManager(run=lambda a, t: (_ for _ in ()).throw(FileNotFoundError("adb")), settle=0)
    assert not m3.on()["ok"]


def test_off_only_when_we_turned_it_on_and_it_is_verified():
    m, f = mgr()
    assert m.off_if_active() is None                       # never touched the headset: nothing to restore
    assert not any("automation_disable" in c for c in f.log)
    m.on()
    r = m.off_if_active()
    assert r["ok"] and not m.sent and f.prox is False and any("automation_disable" in c for c in f.log)


# ------------------------------------------------------------ server integration
def hello_and_report(ws, mode, state="running"):
    ws.send(json.dumps({"t": "hello", "role": "controller"}))
    ws.send(json.dumps({"t": "xr", "state": state, "msg": "immersive-ar", "mode": mode, "tracked": "left+right",
                        "inputs": [], "fps": 72}))


def test_headless_session_arms_the_override_once_and_tells_the_operator(server):
    from piper_teleop.quest_server import QuestServer  # noqa: F401  (fixture builds it)
    srv, system, port = server
    m, f = mgr()
    srv.headless = m
    with connect(f"ws://127.0.0.1:{port}/ws", compression=None) as headset, \
         connect(f"ws://127.0.0.1:{port}/ws", compression=None) as dash:
        dash.send(json.dumps({"t": "hello", "role": "viewer"}))
        hello_and_report(headset, "headless")
        spoken, st = None, None
        for _ in range(200):
            time.sleep(0.02)
            if f.prox:
                break
        assert f.prox, "override was never sent"
        headset.send(json.dumps({"t": "xr", "state": "running", "msg": "again", "mode": "headless", "tracked": "left+right", "inputs": []}))
        for _ in range(80):
            m_ = json.loads(headset.recv(timeout=2))
            if m_.get("t") == "speak":
                spoken = m_["text"]
                break
        assert spoken and "take the headset off" in spoken
        for _ in range(80):
            d = json.loads(dash.recv(timeout=2))
            if d.get("t") == "status" and d.get("headless"):
                st = d["headless"]
                break
        assert st and st["ok"]
    assert sum("prox_close" in c for c in f.log) == 1       # armed exactly once


def test_normal_headset_session_never_touches_the_proximity_sensor(server):
    srv, system, port = server
    m, f = mgr()
    srv.headless = m
    with connect(f"ws://127.0.0.1:{port}/ws", compression=None) as headset:
        hello_and_report(headset, "headset")
        time.sleep(0.5)
    assert not f.log                                         # adb was never even called


def test_override_is_armed_when_the_button_is_pressed_not_after_the_session_is_up(server):
    """On the first real attempt it landed 2 s after the press, when the headset was already off the head."""
    srv, system, port = server
    m, f = mgr()
    srv.headless = m
    with connect(f"ws://127.0.0.1:{port}/ws", compression=None) as headset:
        hello_and_report(headset, "headless", state="starting")          # only 'starting' has been reported
        for _ in range(200):
            time.sleep(0.02)
            if f.prox:
                break
        assert f.prox, "must arm on 'starting'"
        got = []
        t0 = time.time()
        while time.time() - t0 < 3.0 and len([g for g in got if g.get("t") == "haptic"]) < 6:
            try:
                got.append(json.loads(headset.recv(timeout=0.5)))
            except Exception:
                pass
        buzzes = [g for g in got if g.get("t") == "haptic" and g["i"] == 1.0]
        assert len(buzzes) >= 4, "operator must feel a clear armed buzz on both controllers"
    assert sum("prox_close" in c for c in f.log) == 1
