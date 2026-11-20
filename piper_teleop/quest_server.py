"""Meta Quest 3 -> dual Piper teleop server.

    python -m piper_teleop.quest_server --sim                # dry run, no hardware
    python -m piper_teleop.quest_server                      # real arms (config/teleop.yaml)
    python -m piper_teleop.quest_server --tls                # HTTPS/WSS for Wi-Fi use

One port serves: the Quest WebXR page (/), the PC dashboard (/dashboard) and the
WebSocket (/ws).  Poses arrive as 62-byte-per-hand binary frames (see protocol.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import signal
import ssl
import sys
import threading
import time
from http import HTTPStatus
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from websockets.datastructures import Headers
from websockets.http11 import Response

from . import config as cfgmod
from .controller import TeleopArm
from . import paths
from .headless import HeadlessManager
from .protocol import decode_pose
from .system import TeleopSystem

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PAGES = {"/": "index.html", "/index.html": "index.html", "/dashboard": "dashboard.html",
         "/quest.js": "quest.js"}


class QuestServer:
    def __init__(self, system: TeleopSystem, host: str, port: int, ssl_ctx=None, headless: HeadlessManager | None = None):
        self.system = system
        self.host, self.port, self.ssl_ctx = host, port, ssl_ctx
        self.clients: dict = {}           # ws -> info dict
        self.loop: asyncio.AbstractEventLoop | None = None
        self.pose_times: list[float] = []
        self.pose_hz = 0.0
        self.max_gap_ms = 0.0
        self.last_pose_t = 0.0
        self.rtt_ms = None
        self.headless_hint = False
        self.headless = headless            # arms the Quest proximity override for headless sessions
        self.headless_state: dict | None = None
        self._headless_requested = False
        self._logf = None
        self._last_xr_key = {}
        self._last_arm_key = {}
        self._sq_prev = {}
        try:
            lp = paths.log_dir() / "session.log"
            if lp.exists() and lp.stat().st_size > 2_000_000:
                lp.unlink()
            self._logf = open(lp, "a", buffering=1)
            self._log("=== server start ===")
        except OSError:
            self._logf = None

    def _log(self, msg: str):
        if self._logf:
            self._logf.write(time.strftime("%H:%M:%S ") + msg + "\n")

    # -- thread-safe event sink for the control thread -------------------------
    def on_event(self, kind, *args):
        if kind == "gripper_warn":                      # a gripper that sits still far from its command
            print(f"[{args[0]}] WARNING: {args[1]}", flush=True)
            self._log(f"gripper warning {args[0]}: {args[1]}")
            return
        if self.loop is None:
            return
        if kind == "haptic":
            hand, inten, ms = args
            msg = {"t": "haptic", "hand": hand, "i": inten, "ms": ms}
        elif kind == "speak":
            msg = {"t": "speak", "text": args[0]}
            self.loop.call_soon_threadsafe(self._broadcast_viewers, msg)
        elif kind in ("estop", "error", "calibrated", "calib_fail"):
            msg = {"t": "event", "kind": kind, "args": list(args)}
            if kind in ("estop", "error", "calib_fail"):
                for hand in ("left", "right"):
                    self.loop.call_soon_threadsafe(self._broadcast_controllers, {
                        "t": "haptic", "hand": hand, "i": 1.0, "ms": 400})
        else:
            return
        self.loop.call_soon_threadsafe(self._broadcast_controllers, msg)
        self.loop.call_soon_threadsafe(self._broadcast_viewers, {"t": "event", "kind": kind, "args": list(args)})

    def _broadcast_controllers(self, msg):
        data = json.dumps(msg)
        for ws, info in list(self.clients.items()):
            if info["role"] == "controller":
                asyncio.ensure_future(self._safe_send(ws, data))

    def _broadcast_viewers(self, msg):
        data = json.dumps(msg)
        for ws, info in list(self.clients.items()):
            if info["role"] == "viewer":
                asyncio.ensure_future(self._safe_send(ws, data))

    @staticmethod
    async def _safe_send(ws, data):
        try:
            await ws.send(data)
        except Exception:
            pass

    # -- HTTP --------------------------------------------------------------------
    async def process_request(self, connection, request):
        path = request.path.split("?", 1)[0]
        if path == "/ws":
            return None
        name = PAGES.get(path)
        if name is None:
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        body = (WEB / name).read_bytes()
        ctype = mimetypes.guess_type(name)[0] or "text/plain"
        h = Headers([("Content-Type", f"{ctype}; charset=utf-8"), ("Content-Length", str(len(body))),
                     ("Cache-Control", "no-store")])
        return Response(200, "OK", h, body)

    # -- WebSocket ---------------------------------------------------------------
    async def handler(self, ws):
        info = {"role": "controller", "ua": "", "rtt": None}
        self.clients[ws] = info
        self._log(f"client connected from {getattr(ws, 'remote_address', '?')}")
        try:
            async for msg in ws:                 # a Quest that sleeps or is unplugged just ends this loop
                if isinstance(msg, bytes):
                    now = time.perf_counter()
                    try:
                        hands = decode_pose(msg, now)
                    except ValueError:
                        continue
                    for name, s in hands.items():
                        self.system.submit(name, s)
                    self._pose_stats(now)
                else:
                    await self._text(ws, info, msg)
        except ConnectionClosed:
            pass                                 # routine: headset asleep / page closed / cable pulled
        finally:
            self._log(f"client disconnected (role={info['role']}, last xr={info.get('xr', {}).get('state')})")
            self.clients.pop(ws, None)

    def _pose_stats(self, now):
        if self.last_pose_t:
            gap = (now - self.last_pose_t) * 1e3
            self.max_gap_ms = max(self.max_gap_ms * 0.995, gap)
            self.pose_hz = 0.95 * self.pose_hz + 0.05 * (1000.0 / max(gap, 0.1))
        self.last_pose_t = now

    async def _text(self, ws, info, raw):
        try:
            m = json.loads(raw)
        except ValueError:
            return
        t = m.get("t")
        if t == "hello":
            info["role"] = "viewer" if m.get("role") == "viewer" else "controller"
            info["ua"] = str(m.get("ua", ""))[:80]
            info["mode"] = m.get("mode", "")
        elif t == "ping":
            await ws.send(json.dumps({"t": "pong", "c": m.get("c")}))
        elif t == "rtt":
            info["rtt"] = float(m.get("ms", 0))
            self.rtt_ms = info["rtt"]
        elif t == "cmd" and info["role"] == "viewer":
            self.system.request(str(m.get("cmd")), m.get("arm"))
        elif t == "xr" and info["role"] == "controller":
            _prev = info.get("xr")
            info["xr"] = {"state": str(m.get("state", ""))[:20], "msg": str(m.get("msg", ""))[:200],
                          "tracked": str(m.get("tracked", ""))[:40], "vis": str(m.get("vis", ""))[:20],
                          "fps": m.get("fps"), "sent": m.get("sent"), "blend": str(m.get("blend", ""))[:20],
                          "mode": str(m.get("mode", ""))[:12],
                          "inputs": [{k: (str(v)[:60] if k == "p" else v) for k, v in i.items()}
                                     for i in (m.get("inputs") or [])[:6] if isinstance(i, dict)]}
            x = info["xr"]
            self._maybe_headless(x)
            key = (x["state"], x["msg"], x["tracked"], x["vis"], x["blend"], json.dumps(x["inputs"], sort_keys=True))
            if self._last_xr_key.get(id(ws)) != key:        # log changes only (fps/sent alone would be noise)
                self._last_xr_key[id(ws)] = key
                self._log(f"headset page: state={x['state']} msg={x['msg']!r} tracked={x['tracked']!r} vis={x['vis']!r} "
                          f"blend={x['blend']!r} fps={x['fps']} inputs={x['inputs']}")
        elif t == "estop":          # allowed from any client, incl. the headset page
            self.system.request("estop")

    # -- headless: arm the proximity override once a headless VR session is actually running -------------
    def _maybe_headless(self, x: dict):
        if self.headless is None or self.loop is None:
            return
        if x.get("state") in ("ended", "error", "idle"):
            self._headless_requested = False                   # a new session will re-arm it
            return
        # Arm as soon as the operator PRESSES the button ("starting"), not when the session is already up: on the first
        # real attempt the override landed 2 s after the press, by which time the headset was already off the head.
        if x.get("state") in ("starting", "running") and x.get("mode") == "headless" and not self._headless_requested:
            self._headless_requested = True
            self.loop.run_in_executor(None, self._headless_on)

    def _headless_on(self):
        res = self.headless.on()
        self.headless_state = {"requested": True, **res}
        self._log(f"headless: {res}")
        if res["ok"]:
            for _ in range(3):                                # a clear triple buzz on both controllers: "armed"
                for h in ("left", "right"):
                    self.on_event("haptic", h, 1.0, 220)
                time.sleep(0.4)
        if res["ok"] and not res["note"]:
            self.on_event("speak", "Headless mode is on. You can take the headset off now. Stand it upright facing the room, "
                                   "so its cameras can see the controllers.")
        elif res["ok"]:
            self.on_event("speak", "Headless mode is on, but the headset reports a problem. Check the dashboard.")
        else:
            self.on_event("speak", "Headless mode could not be turned on. Check the dashboard.")
            self.on_event("haptic", "left", 1.0, 300)
            self.on_event("haptic", "right", 1.0, 300)

    async def status_task(self):
        n = 0
        while True:
            await asyncio.sleep(0.05)
            n += 1
            snap = self.system.snapshot()
            snap["net"] = {"pose_hz": round(self.pose_hz, 1), "max_gap_ms": round(self.max_gap_ms, 1),
                           "rtt_ms": self.rtt_ms,
                           "stream_age_ms": round((time.perf_counter() - self.last_pose_t) * 1e3)
                           if self.last_pose_t else None,
                           "clients": [c["role"] for c in self.clients.values()],
                           "quest": [c.get("xr") or {"state": "no report yet", "msg": ""}
                                     for c in self.clients.values() if c["role"] == "controller"]}
            now = time.perf_counter()
            snap["hands"] = {h: {"age_ms": round((now - s.t_recv) * 1e3), "tracked": s.tracked, "emulated": s.emulated,
                                 "squeeze": round(s.squeeze, 2), "trigger": round(s.trigger, 2),
                                 "a": s.btn_a, "b": s.btn_b, "stick_click": s.stick_click}
                             for h, s in dict(self.system.samples).items()}
            for h, v in snap["hands"].items():              # log GRIP going down/up, so "grip does nothing" is diagnosable
                down = v["squeeze"] > 0.6
                if self._sq_prev.get(h) != down:
                    self._sq_prev[h] = down
                    self._log(f"{h} GRIP {'PRESSED' if down else 'released'} (tracked={v['tracked']} emulated={v['emulated']})")
            for name, a in snap["arms"].items():
                k = (a["state"], a["msg"])
                if self._last_arm_key.get(name) != k:
                    self._last_arm_key[name] = k
                    self._log(f"arm {name}: {a['state']} - {a['msg']}")
            snap["headless"] = self.headless_state
            snap["t"] = "status"
            full = json.dumps(snap)
            small = None
            for ws, info in list(self.clients.items()):
                if info["role"] == "viewer":
                    asyncio.ensure_future(self._safe_send(ws, full))
                elif n % 4 == 0:   # controllers get a light 5 Hz summary (for page colour/text)
                    if small is None:
                        small = json.dumps({"t": "status", "calib": snap["calibration"], "arms": {
                            k: {"state": v["state"], "msg": v["msg"], "fine": v["fine"]}
                            for k, v in snap["arms"].items()}})
                    asyncio.ensure_future(self._safe_send(ws, small))

    async def run(self):
        self.loop = asyncio.get_running_loop()
        async with serve(self.handler, self.host, self.port, process_request=self.process_request,
                         ssl=self.ssl_ctx, compression=None, ping_interval=10, ping_timeout=30,
                         max_size=4096, max_queue=4):
            asyncio.ensure_future(self.status_task())
            scheme = "https" if self.ssl_ctx else "http"
            print(f"[server] listening on {scheme}://{self.host}:{self.port}  "
                  f"(dashboard: {scheme}://localhost:{self.port}/dashboard)", flush=True)
            await asyncio.Future()


def build_system(raw: dict, force_sim: bool, only=None) -> TeleopSystem:
    arms = []
    yaw = raw.get("control", {}).get("frame_yaw_deg", 0.0)
    for name, section in raw["arms"].items():
        if only and name not in only:
            continue
        backend = cfgmod.make_backend(section, raw, force_sim)
        backend.connect()
        acfg = cfgmod.arm_config(name, raw, section)
        if not backend.enable(gripper=acfg.gripper_enabled):
            raise SystemExit(f"[{name}] could not enable arm on {section.get('can')}: "
                             f"{getattr(backend, 'error', '') or 'enable failed'}\n"
                             f"See the setup guide, chapter 12 (Troubleshooting).")
        for note in getattr(backend, "notes", []):
            print(f"[{name}] {note}", flush=True)
        gs = getattr(backend, "gripper_status", lambda: None)()
        if gs is not None and gs["error"]:
            print(f"[{name}] WARNING: the gripper reports a fault (overheating / over-current / low voltage / sensor).", flush=True)
        arms.append(TeleopArm(acfg, backend, frame_yaw_deg=yaw))
    ctl = raw.get("control", {})
    return TeleopSystem(arms, rate_hz=ctl.get("rate_hz", 200), hard_estop=ctl.get("estop_mode", "freeze") == "firmware",
                        auto_calibrate=raw.get("calibration", {}).get("auto", True))


def _port_free(host: str, port: int) -> bool:
    import socket
    with socket.socket() as sk:
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sk.bind(("" if host in ("0.0.0.0", "") else host, port))
            return True
        except OSError:
            return False


def _tls(raw: dict, args) -> ssl.SSLContext | None:
    if not args.tls:
        return None
    t = raw.get("transport", {})
    cert = ROOT / t.get("tls_cert", "certs/cert.pem")
    key = ROOT / t.get("tls_key", "certs/key.pem")
    if not cert.exists():
        raise SystemExit(f"TLS requested but {cert} missing - run scripts/make_cert.sh first")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    return ctx


KEYS = {" ": "estop", "e": "estop", "r": "resume", "h": "home", "p": "park", "c": "calibrate", "x": "calibrate_cancel"}


def console_command(ch: str):
    """One keypress in the server terminal -> ('cmd', name) | ('quit', None) | None.  Space = FREEZE."""
    if ch == "Q":                    # capital Q only: a stray 'q' must never stop a live session
        return ("quit", None)
    name = KEYS.get(ch)
    return ("cmd", name) if name else None


def _cbreak_stdin():
    """Make single keypresses (no Enter) readable; returns a function that restores the terminal."""
    if not sys.stdin.isatty():
        return lambda: None
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)                # Ctrl-C still raises SIGINT
    return lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _stdin_commands(system: TeleopSystem, stop: threading.Event):
    if not sys.stdin.isatty():
        return
    import os
    import select
    print("[console] SPACE = FREEZE all arms (software hold, no power cut)   r=resume  h=home  p=park  "
          "c=axis calibration  x=cancel it  Q=quit+park", flush=True)
    while not stop.is_set():
        if not select.select([sys.stdin], [], [], 0.1)[0]:
            continue
        ch = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        act = console_command(ch)
        if act is None:
            continue
        if act[0] == "quit":
            stop.set()
            return
        system.request(act[1])
        print(f"[console] {'FREEZE' if act[1] == 'estop' else act[1]}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--sim", action="store_true", help="simulated arms, no CAN hardware needed")
    ap.add_argument("--arms", nargs="*", help="only these arm names (e.g. --arms left)")
    ap.add_argument("--port", type=int)
    ap.add_argument("--host")
    ap.add_argument("--tls", action="store_true", help="serve HTTPS/WSS (needed for Wi-Fi use)")
    ap.add_argument("--usb", action="store_true",
                    help="once listening, run scripts/quest_adb.sh usb PORT (adb reverse + open the page in the Quest)")
    ap.add_argument("--no-headless-auto", action="store_true",
                    help="do not arm the Quest proximity override by itself for headless sessions")
    ap.add_argument("--no-park", action="store_true", help="on exit leave arms enabled where they are")
    ap.add_argument("--release-on-exit", action="store_true",
                    help="after parking, DISABLE the motors (the wrist droops). Default: leave them holding")
    ap.add_argument("--yes", action="store_true", help="do not ask before moving to home")
    args = ap.parse_args(argv)

    raw = cfgmod.load(args.config)
    tr = raw.get("transport", {})
    host = args.host or tr.get("host", "0.0.0.0")
    port = args.port or tr.get("port", 8788)

    if not _port_free(host, port):          # BEFORE touching any arm: a server that cannot listen is useless
        raise SystemExit(f"port {port} is already in use by another program.  Pick a free one:  "
                         f"--port N   (or transport.port in config/teleop.yaml).  Nothing was started and no arm was touched.\n"
                         f"See who has it:  ss -ltnp | grep :{port}")
    system = build_system(raw, args.sim, args.arms)
    import shutil
    hm = HeadlessManager() if (shutil.which("adb") and not args.no_headless_auto) else None
    server = QuestServer(system, host, port, _tls(raw, args), headless=hm)
    system.on_event = server.on_event
    system.start()

    if not args.sim and not args.yes:
        input("\n>>> Clear the workspace. Press Enter to move the arm(s) to the HOME pose "
              "(Ctrl-C to abort)... ")
    system.request("home")

    stop = threading.Event()
    restore_tty = _cbreak_stdin()
    threading.Thread(target=_stdin_commands, args=(system, stop), daemon=True).start()

    async def main_async():
        task = asyncio.ensure_future(server.run())
        loop = asyncio.get_running_loop()

        if args.usb:
            def _usb():
                import subprocess
                time.sleep(1.0)               # the server is listening by now
                r = subprocess.run([str(ROOT / "scripts" / "quest_adb.sh"), "usb", str(port)],
                                   capture_output=True, text=True)
                print("[usb] " + (r.stdout + r.stderr).strip().replace("\n", "\n[usb] "), flush=True)
            loop.run_in_executor(None, _usb)

        def _server_died(t):                 # never keep running "headless" without noticing
            if not t.cancelled() and t.exception() is not None:
                print(f"\n[server] FAILED: {t.exception()!r} - shutting down", flush=True)
                stop.set()
        task.add_done_callback(_server_died)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        while not stop.is_set():
            await asyncio.sleep(0.1)
        task.cancel()

    try:
        asyncio.run(main_async())
    finally:
        restore_tty()
        if hm is not None:
            res = hm.off_if_active()
            if res is not None:
                print("[headless] headset proximity behaviour " + ("restored" if res["ok"] else "NOT confirmed restored: " + res["note"]), flush=True)
        print("\n[server] shutting down" + ("" if args.no_park else " - parking arms (Ctrl-C again to skip)"),
              flush=True)
        try:
            system.shutdown(park=not args.no_park, release=args.release_on_exit)
        except KeyboardInterrupt:
            system.shutdown(park=False)


if __name__ == "__main__":
    main()
