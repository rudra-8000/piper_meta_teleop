"""The recording session: which keys do what, the 30 Hz frame clock, and what happens to each episode.

States and keys (all keys work while the arms keep teleoperating; only what is WRITTEN changes):

    WAIT     live teleop, nothing recorded.          Enter / Space / s : start episode      q : quit
    REC      recording                               n / Enter / Space : finish episode     p : pause
                                                     r : discard it and start over          q : finish, then quit
    PAUSED   recording is paused (frames are not     p / Space : resume                     n : finish
             written; the episode continues)         r : discard it                         q : finish, then quit
    REVIEW   the finished episode is NOT saved yet   Enter / k : keep it (writes it)        r : retake (discard, record again)
                                                     d : discard, don't retake              q : keep it and quit

After a kept or discarded episode the session goes back to WAIT (reset the scene, press Enter).  With ``reset_time_s`` the
next episode starts by itself after that many seconds, with ``episode_time_s`` an episode ends by itself (and goes to
REVIEW, or is saved at once with ``review=False``).  ``num_episodes`` stops the session after that many KEPT episodes.

The session is independent of the robot: the runner calls ``key(ch)`` for each key press and ``tick(now, sample)`` as
often as it likes (200 Hz is fine); ``tick`` writes a frame only when the next 1/fps boundary has passed.
"""
from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty

WAIT, REC, PAUSED, REVIEW, DONE = "WAIT", "REC", "PAUSED", "REVIEW", "DONE"

HELP = ("keys: Enter=start/finish  p=pause/resume  r=discard & retake  n=finish  k=keep  d=discard  q=quit  h=help")


class KeyReader:
    """Non-blocking single-key reads from the terminal (no echo, Ctrl-C still works).  Does nothing without a tty."""

    def __init__(self, fd=None):
        self.fd = sys.stdin.fileno() if fd is None else fd
        self._old = None
        self.active = False

    def start(self) -> bool:
        if not os.isatty(self.fd):
            return False
        self._old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.active = True
        return True

    def poll(self) -> list[str]:
        keys: list[str] = []
        while self.active and select.select([self.fd], [], [], 0)[0]:
            data = os.read(self.fd, 64).decode("utf-8", "ignore")
            if data.startswith("\x1b") and len(data) > 1:        # arrow / function key: ignore the whole sequence
                continue
            keys.extend(ch for ch in data if ch != "\x1b")
        return keys

    def stop(self):
        if self._old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            self._old = None
        self.active = False


class RecordingSession:
    def __init__(self, writer, cameras: dict, *, task: str, fps: int = 30, episode_time_s: float = 0.0,
                 reset_time_s: float = 0.0, num_episodes: int = 0, review: bool = True, out=print, status_out=None,
                 clock=time.perf_counter, on_wait=None, ready=None, info=None):
        self.w, self.cams, self.task, self.fps = writer, cameras, task, int(fps)
        self.episode_time_s, self.reset_time_s, self.num_episodes, self.review = episode_time_s, reset_time_s, num_episodes, review
        self.out, self.clock = out, clock
        self.on_wait = on_wait                 # called each time the session goes back to WAIT (e.g. send the robots home)
        self.ready = ready or (lambda: True)   # an episode may only start (by timer) while this is true
        self.info = info or (lambda: "")       # extra text for the WAIT status line (what the robots are doing)
        self.status_out = status_out or (lambda line: print("\r" + line.ljust(110), end="", flush=True))
        self.state = WAIT
        self.kept = 0                          # episodes kept this session
        self.quit_after = False
        self.dropped = 0                       # frame slots with no sample / that the loop was too slow for (this episode)
        self.stale = 0                         # frames taken from a camera that had not updated for > 0.15 s
        self._next = 0.0
        self._wait_since = None                # set after an episode, for reset_time_s
        self._last_status = 0.0
        self.log: list[str] = []
        self.say(f"dataset: {self.w.root}  ({self.w.fmt}, {self.fps} fps, {self.w.num_episodes} episodes already)")
        self.say(f"task: {task!r}")
        self.say(HELP)
        self._prompt_wait()

    # ------------------------------------------------------------------ output
    def say(self, msg: str):
        self.log.append(msg)
        self.out(msg if self.out is not print else "\r\x1b[K" + msg)       # start a fresh line over the live status line

    def _prompt_wait(self):
        n = self.w.num_episodes
        self.say(f"ready for episode {n}: reset the scene, then press Enter to start recording")

    @property
    def done(self) -> bool:
        return self.state == DONE

    def status_line(self) -> str:
        n = self.w.frames_in_episode
        if self.state in (REC, PAUSED):
            extra = f"  dropped {self.dropped}" if self.dropped else ""
            extra += f"  STALE CAMERA x{self.stale}" if self.stale else ""
            return f"[{self.state:6s}] episode {self.w.num_episodes}  {n / self.fps:6.1f} s  {n:5d} frames{extra}"
        if self.state == REVIEW:
            return f"[REVIEW] episode {self.w.num_episodes}: {n} frames ({n / self.fps:.1f} s)   Enter=keep  r=retake  d=discard  q=keep & quit"
        extra = self.info()
        return f"[{self.state:6s}] kept {self.kept} this session ({self.w.num_episodes} in the dataset)" + (f"  | {extra}" if extra else "")

    def maybe_status(self, now: float, interval: float = 0.25):
        """Print the live status line (overwrites itself) at most every `interval` s."""
        if now - self._last_status >= interval:
            self._last_status = now
            self.status_out(self.status_line())

    # -------------------------------------------------------------------- keys
    def key(self, ch: str):
        ch = {"\r": "\n", " ": " "}.get(ch, ch).lower()
        if ch == "h":
            self.say(HELP)
            return
        s = self.state
        if s == WAIT:
            if ch in ("\n", " ", "s"):
                self.start_episode()
            elif ch == "q":
                self.stop("quit")
        elif s == REC:
            if ch in ("n", "\n", " "):
                self.finish()
            elif ch == "p":
                self.state = PAUSED
                self.say("paused - nothing is written; press p to resume")
            elif ch == "r":
                self.discard("discarded - press Enter to start it again")
            elif ch == "q":
                self.quit_after = True
                self.finish()
        elif s == PAUSED:
            if ch in ("p", " "):
                self.state = REC
                self._next = self.clock()
                self.say("resumed")
            elif ch == "n":
                self.finish()
            elif ch == "r":
                self.discard("discarded - press Enter to start it again")
            elif ch == "q":
                self.quit_after = True
                self.finish()
        elif s == REVIEW:
            if ch in ("\n", "k"):
                self.keep()
            elif ch == "r":
                self.discard("retake: press Enter to record it again")
            elif ch == "d":
                self.discard("discarded")
            elif ch == "q":
                self.quit_after = True
                self.keep()

    # ----------------------------------------------------------------- actions
    def start_episode(self):
        self.w.begin_episode(self.task)
        self.dropped = self.stale = 0
        self._next = self.clock()
        self.state = REC
        self.say(f"recording episode {self.w.num_episodes} ...")

    def finish(self):
        n = self.w.frames_in_episode
        if n == 0:
            self.discard("nothing was recorded - discarded")
            return
        self.state = REVIEW
        self.say("\n" + self.status_line())
        if not self.review:
            self.keep()

    def keep(self):
        idx = self.w.save_episode()
        self.kept += 1
        self.say(f"saved episode {idx} ({self.w.total_frames} frames in the dataset)")
        self._back_to_wait()

    def discard(self, msg: str):
        if self.w.in_episode:
            self.w.discard_episode()
        self.say(msg)
        self._back_to_wait(prompt=False)

    def _back_to_wait(self, prompt: bool = True):
        self.state = WAIT
        self._wait_since = self.clock()
        if self.quit_after or (self.num_episodes and self.kept >= self.num_episodes):
            self.stop("done" if not self.quit_after else "quit")
        else:
            if self.on_wait is not None:
                self.on_wait()
            if prompt:
                self._prompt_wait()

    def stop(self, why: str):
        if self.w.in_episode:
            self.w.discard_episode()
            self.say("the unfinished episode was discarded")
        self.state = DONE
        self.say(f"stopping ({why}): {self.kept} episode(s) kept this session")

    # -------------------------------------------------------------------- tick
    def tick(self, now: float, sample):
        """`sample` = (state_vector, action_vector) for this instant, or None when the robot has nothing valid."""
        if self.state == WAIT:
            if self.reset_time_s and self._wait_since is not None and now - self._wait_since >= self.reset_time_s and self.ready():
                self.start_episode()
            return
        if self.state != REC or now < self._next:
            return
        period = 1.0 / self.fps
        late = int((now - self._next) / period)
        self._next += period * (late + 1)               # never burst to catch up: one frame per slot, slots may be skipped
        if late:
            self.dropped += late
        if sample is None:
            self.dropped += 1
            return
        images = {}
        for name, cam in self.cams.items():
            images[name] = cam.read()
            if cam.frame_age() > 0.15:
                self.stale += 1
        self.w.add_frame(sample[0], sample[1], images)
        if self.episode_time_s and self.w.frames_in_episode >= self.episode_time_s * self.fps:
            self.say("episode time reached")
            self.finish()

    def close(self):
        if self.w.in_episode:
            self.w.discard_episode()
