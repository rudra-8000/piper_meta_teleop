import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from piper_teleop.recording.cameras import FakeCamera
from piper_teleop.recording.session import DONE, PAUSED, REC, REVIEW, WAIT, KeyReader, RecordingSession
from piper_teleop.recording.writer import LeRobotWriter

SN = ["a", "b", "c"]


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def rig(tmp_path, fmt="v3.0", cams=True, **kw):
    cam = {"wrist": FakeCamera("wrist", 64, 48)} if cams else {}
    w = LeRobotWriter(tmp_path / "ds", fps=30, state_names=SN, action_names=SN,
                      cameras={"wrist": (48, 64)} if cams else {}, fmt=fmt)
    clk = Clock()
    s = RecordingSession(w, cam, task="stack the blocks", fps=30, out=lambda *a, **k: None, status_out=lambda l: None, clock=clk, **kw)
    return s, w, clk


def run(s, clk, seconds, sample=lambda t: (np.array([t, 0, 0]), np.array([0, t, 0])), dt=0.005):
    end = clk.t + seconds
    while clk.t < end:
        clk.t += dt
        s.tick(clk.t, sample(clk.t))


def test_records_at_30_hz_from_a_200_hz_loop_and_waits_for_review(tmp_path):
    s, w, clk = rig(tmp_path)
    assert s.state == WAIT
    run(s, clk, 1.0)
    assert w.frames_in_episode == 0                                 # WAIT records nothing
    s.key("\n")
    assert s.state == REC
    run(s, clk, 2.0)
    assert w.frames_in_episode in range(59, 62)                     # 2 s at 30 Hz
    s.key("n")
    assert s.state == REVIEW and w.num_episodes == 0                # finished but NOT saved yet
    s.key("k")
    assert s.state == WAIT and w.num_episodes == 1 and s.kept == 1
    assert pq.read_table(tmp_path / "ds/data/chunk-000/file-000.parquet").num_rows in range(59, 62)


def test_pause_stops_writing_and_resume_continues_the_same_episode(tmp_path):
    s, w, clk = rig(tmp_path)
    s.key("s"); run(s, clk, 1.0)
    n1 = w.frames_in_episode
    s.key("p")
    assert s.state == PAUSED
    run(s, clk, 3.0)
    assert w.frames_in_episode == n1                                # nothing written while paused
    s.key("p")
    assert s.state == REC
    run(s, clk, 1.0)
    assert w.frames_in_episode in range(n1 + 28, n1 + 32)           # resumed without a burst to "catch up"
    s.key("n"); s.key("k")
    assert w.num_episodes == 1                                      # one episode, not two
    assert s.dropped == 0


def test_discard_while_recording_and_retake_from_review(tmp_path):
    s, w, clk = rig(tmp_path)
    s.key("\n"); run(s, clk, 1.0)
    s.key("r")                                                      # throw away the take in progress
    assert s.state == WAIT and w.num_episodes == 0 and not w.in_episode
    s.key("\n"); run(s, clk, 1.0); s.key("n")
    assert s.state == REVIEW
    s.key("r")                                                      # bad take: retake
    assert s.state == WAIT and w.num_episodes == 0
    s.key("\n"); run(s, clk, 0.5); s.key("n"); s.key("\n")          # Enter in REVIEW keeps it
    assert w.num_episodes == 1 and w.frames_in_episode == 0
    info = json.loads((tmp_path / "ds/meta/info.json").read_text())
    assert info["total_episodes"] == 1 and info["total_frames"] in range(13, 17)      # only the kept take is in the dataset


def test_discard_from_review_does_not_retake(tmp_path):
    s, w, clk = rig(tmp_path)
    s.key("\n"); run(s, clk, 0.5); s.key("n"); s.key("d")
    assert s.state == WAIT and w.num_episodes == 0


def test_quit_while_recording_goes_through_review_then_stops(tmp_path):
    s, w, clk = rig(tmp_path)
    s.key("\n"); run(s, clk, 1.0)
    s.key("q")
    assert s.state == REVIEW and s.quit_after                       # the take is not thrown away silently
    s.key("k")
    assert s.state == DONE and w.num_episodes == 1
    s2, w2, clk2 = rig(tmp_path / "x")
    s2.key("\n"); run(s2, clk2, 1.0); s2.key("q"); s2.key("d")
    assert s2.state == DONE and w2.num_episodes == 0


def test_quit_while_waiting(tmp_path):
    s, w, clk = rig(tmp_path)
    s.key("q")
    assert s.done


def test_episode_time_ends_the_episode_by_itself_and_no_review_saves_it(tmp_path):
    s, w, clk = rig(tmp_path, episode_time_s=1.0, review=False)
    s.key("\n"); run(s, clk, 2.0)
    assert w.num_episodes == 1 and s.state == WAIT
    assert pq.read_table(tmp_path / "ds/data/chunk-000/file-000.parquet").num_rows == 30


def test_reset_time_starts_the_next_episode_automatically_and_num_episodes_stops(tmp_path):
    s, w, clk = rig(tmp_path, episode_time_s=0.5, reset_time_s=1.0, num_episodes=2, review=False)
    s.key("\n"); run(s, clk, 0.7)
    assert w.num_episodes == 1 and s.state == WAIT
    run(s, clk, 0.5)
    assert s.state == WAIT                                          # still resetting
    run(s, clk, 1.0)
    assert s.state in (REC, DONE)
    run(s, clk, 1.0)
    assert s.done and w.num_episodes == 2 and s.kept == 2


def test_a_missing_sample_drops_that_frame_and_a_slow_loop_does_not_burst(tmp_path):
    s, w, clk = rig(tmp_path)
    s.key("\n")
    run(s, clk, 0.5, sample=lambda t: None)
    assert w.frames_in_episode == 0 and s.dropped >= 14
    s.tick(clk.t + 1.0, (np.zeros(3), np.zeros(3)))                 # the loop stalled for a second
    clk.t += 1.0
    assert w.frames_in_episode == 1                                 # one frame, not thirty
    assert s.dropped >= 14 + 29


def test_a_stale_camera_is_counted(tmp_path):
    s, w, clk = rig(tmp_path)
    s.cams["wrist"].frame_age = lambda: 0.5
    s.key("\n"); run(s, clk, 0.3)
    assert s.stale > 5 and "STALE" in s.status_line()


def test_keys_are_ignored_in_states_where_they_mean_nothing(tmp_path):
    s, w, clk = rig(tmp_path)
    for k in ("n", "p", "k", "d", "r"):
        s.key(k)
    assert s.state == WAIT and w.num_episodes == 0
    s.key("\n"); s.key("k")
    assert s.state == REC


@pytest.mark.parametrize("fmt", ["v2.1", "v3.0"])
def test_a_whole_session_in_both_formats(tmp_path, fmt):
    s, w, clk = rig(tmp_path, fmt=fmt)
    for n in (1.0, 0.7, 1.2):
        s.key("\n"); run(s, clk, n); s.key("n"); s.key("k")
    s.key("q")
    assert s.done and w.num_episodes == 3
    info = json.loads((tmp_path / "ds/meta/info.json").read_text())
    assert info["codebase_version"] == fmt and info["total_episodes"] == 3


def test_key_reader_without_a_terminal_is_inert(tmp_path):
    f = open(tmp_path / "notatty", "w+")
    k = KeyReader(f.fileno())
    assert k.start() is False and k.poll() == []
    k.stop()
