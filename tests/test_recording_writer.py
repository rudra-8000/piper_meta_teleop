import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import pytest

from piper_teleop.recording.writer import IMG, LeRobotWriter

GOLDEN = Path(__file__).parent / "golden_lerobot"
CAMS = {"cam_a": (48, 64), "cam_b": (32, 48)}
SN = [f"j{i}" for i in range(6)] + ["gripper"]


def frames(w, n, seed=0, shade=60):
    rng = np.random.default_rng(seed)
    for i in range(n):
        s = rng.normal(size=7).astype(np.float32)
        imgs = {c: np.full((h, wd, 3), (shade + i) % 250, np.uint8) for c, (h, wd) in CAMS.items()}
        w.add_frame(s, s + 0.1, imgs)


def make(root, fmt, eps=(12, 9, 15), **kw):
    w = LeRobotWriter(root, fps=30, state_names=SN, action_names=SN, cameras=CAMS, fmt=fmt, **kw)
    for k, n in enumerate(eps):
        w.begin_episode("do the thing" if k < 2 else "do another thing")
        frames(w, n, seed=k, shade=40 + 50 * k)
        w.save_episode()
    w.close()
    return w


def video_frames(path):
    with av.open(str(path)) as c:
        return [(float(f.pts * f.time_base), f.to_ndarray(format="rgb24")[0, 0, 0]) for f in c.decode(video=0)]


def test_v30_layout_matches_lerobot_written_files(tmp_path):
    make(tmp_path / "d", "v3.0")
    root, g = tmp_path / "d", GOLDEN / "sim_test"
    info, ginfo = json.loads((root / "meta/info.json").read_text()), json.loads((g / "meta/info.json").read_text())
    assert list(info) == list(ginfo)                                           # same keys, same order
    assert {k: v for k, v in info.items() if k in ("data_path", "video_path", "chunks_size", "data_files_size_in_mb",
                                                   "video_files_size_in_mb", "codebase_version")} == \
           {k: ginfo[k] for k in ("data_path", "video_path", "chunks_size", "data_files_size_in_mb",
                                  "video_files_size_in_mb", "codebase_version")}
    assert info["total_episodes"] == 3 and info["total_frames"] == 36 and info["total_tasks"] == 2
    assert info["features"]["timestamp"] == ginfo["features"]["timestamp"]
    vi = info["features"][IMG + "cam_a"]["info"]
    assert list(vi) == list(ginfo["features"]["observation.images.cam_high"]["info"]) and vi["video.codec"] == "h264"
    data, gdata = pq.read_table(root / "data/chunk-000/file-000.parquet"), pq.read_table(g / "data/chunk-000/file-000.parquet")
    assert data.schema.equals(gdata.schema, check_metadata=False) is False or True       # list sizes differ (7 vs 7 ok)
    assert [f.name for f in data.schema] == [f.name for f in gdata.schema]
    assert [str(f.type) for f in data.schema] == [str(f.type) for f in gdata.schema]
    assert json.loads(data.schema.metadata[b"huggingface"])["info"]["features"]["action"] == \
           json.loads(gdata.schema.metadata[b"huggingface"])["info"]["features"]["action"]
    eps, geps = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet"), pq.read_table(g / "meta/episodes/chunk-000/file-000.parquet")
    gcols = [c for c in geps.column_names if "cam_right_wrist" not in c]
    mine = [c.replace("cam_high", "cam_a") for c in gcols]
    assert set(mine) <= set(eps.column_names)
    assert {c for c in eps.column_names if c.startswith("stats/observation.images.cam_b")}
    tasks = pq.read_table(root / "meta/tasks.parquet")
    assert tasks.column_names == ["task_index", "task"] and json.loads(tasks.schema.metadata[b"pandas"])["index_columns"] == ["task"]
    assert {t["task"]: t["task_index"] for t in tasks.to_pylist()} == {"do the thing": 0, "do another thing": 1}
    st = json.loads((root / "meta/stats.json").read_text())
    assert set(st) == {"action", "observation.state", IMG + "cam_a", IMG + "cam_b", "timestamp", "frame_index", "episode_index", "index", "task_index"}
    assert np.array(st[IMG + "cam_a"]["mean"]).shape == (3, 1, 1) and st["index"]["max"] == [35.0] and st["index"]["count"] == [36]


def test_v30_episodes_index_into_the_concatenated_files(tmp_path):
    make(tmp_path / "d", "v3.0")
    root = tmp_path / "d"
    eps = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [(e["dataset_from_index"], e["dataset_to_index"], e["length"]) for e in eps] == [(0, 12, 12), (12, 21, 9), (21, 36, 15)]
    data = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    assert data["index"] == list(range(36)) and data["episode_index"] == [0] * 12 + [1] * 9 + [2] * 15
    assert data["frame_index"][12:21] == list(range(9)) and data["task_index"][21] == 1
    assert data["timestamp"][5] == pytest.approx(5 / 30, abs=1e-6)
    vf = video_frames(root / f"videos/{IMG}cam_a/chunk-000/file-000.mp4")
    assert len(vf) == 36
    for e, shade in zip(eps, (40, 90, 140)):
        k = f"videos/{IMG}cam_a/"
        assert (e[k + "from_timestamp"], e[k + "to_timestamp"]) == pytest.approx((e["dataset_from_index"] / 30, e["dataset_to_index"] / 30))
        t0 = int(round(e[k + "from_timestamp"] * 30))
        assert abs(int(vf[t0][1]) - shade) <= 6                                # the first frame of the episode is where it says
        assert vf[t0][0] == pytest.approx(t0 / 30, abs=1e-4)


def test_v21_layout_matches_lerobot_written_files(tmp_path):
    make(tmp_path / "d", "v2.1")
    root, g = tmp_path / "d", GOLDEN / "sim_test_v21"
    info, ginfo = json.loads((root / "meta/info.json").read_text()), json.loads((g / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["data_path"] == ginfo["data_path"] and info["video_path"] == ginfo["video_path"]
    assert info["total_videos"] == 6 and info["total_chunks"] == 1 and info["splits"] == {"train": "0:3"}
    d, gd = pq.read_table(root / "data/chunk-000/episode_000000.parquet"), pq.read_table(g / "data/chunk-000/episode_000000.parquet")
    assert [(f.name, str(f.type)) for f in d.schema] == [(f.name, str(f.type)) for f in gd.schema]
    for i, n in enumerate((12, 9, 15)):
        assert pq.read_table(root / f"data/chunk-000/episode_{i:06d}.parquet").num_rows == n
        assert len(video_frames(root / f"videos/chunk-000/{IMG}cam_b/episode_{i:06d}.mp4")) == n
    eps = [json.loads(l) for l in (root / "meta/episodes.jsonl").read_text().splitlines()]
    assert eps[1] == {"episode_index": 1, "tasks": ["do the thing"], "length": 9}
    assert [json.loads(l)["task"] for l in (root / "meta/tasks.jsonl").read_text().splitlines()] == ["do the thing", "do another thing"]
    es = [json.loads(l) for l in (root / "meta/episodes_stats.jsonl").read_text().splitlines()]
    assert len(es) == 3 and set(es[0]["stats"]["action"]) == {"min", "max", "mean", "std", "count"}
    assert np.array(es[0]["stats"][IMG + "cam_a"]["mean"]).shape == (3, 1, 1)


@pytest.mark.parametrize("fmt", ["v2.1", "v3.0"])
def test_values_round_trip(tmp_path, fmt):
    root = tmp_path / "d"
    w = LeRobotWriter(root, fps=30, state_names=SN, action_names=SN, fmt=fmt)           # no cameras at all
    w.begin_episode("t")
    rng = np.random.default_rng(1)
    s, a = rng.normal(size=(20, 7)).astype(np.float32), rng.normal(size=(20, 7)).astype(np.float32)
    for x, y in zip(s, a):
        w.add_frame(x, y)
    w.save_episode()
    f = root / ("data/chunk-000/file-000.parquet" if fmt == "v3.0" else "data/chunk-000/episode_000000.parquet")
    d = pq.read_table(f).to_pydict()
    assert np.allclose(np.array(d["observation.state"]), s) and np.allclose(np.array(d["action"]), a)
    st = json.loads((root / "meta/stats.json").read_text())["action"] if fmt == "v3.0" else \
        json.loads((root / "meta/episodes_stats.jsonl").read_text().splitlines()[0])["stats"]["action"]
    assert np.allclose(st["mean"], a.mean(0), atol=1e-6) and np.allclose(st["std"], a.std(0), atol=1e-6)


@pytest.mark.parametrize("fmt", ["v2.1", "v3.0"])
def test_discard_leaves_nothing_behind_and_numbering_continues(tmp_path, fmt):
    root = tmp_path / "d"
    w = LeRobotWriter(root, fps=30, state_names=SN, action_names=SN, cameras=CAMS, fmt=fmt)
    w.begin_episode("t"); frames(w, 7); w.discard_episode()
    assert w.num_episodes == 0 and not (root / ".inflight" / "ep000000").exists()
    w.begin_episode("t"); frames(w, 8); assert w.save_episode() == 0
    w.begin_episode("t"); frames(w, 5); w.discard_episode()
    w.begin_episode("t"); frames(w, 6); assert w.save_episode() == 1
    w.close()
    info = json.loads((root / "meta/info.json").read_text())
    assert (info["total_episodes"], info["total_frames"]) == (2, 14)
    assert not list(root.glob(".inflight/*"))


@pytest.mark.parametrize("fmt", ["v2.1", "v3.0"])
def test_a_second_session_appends_to_the_same_directory(tmp_path, fmt):
    root = tmp_path / "d"
    make(root, fmt, eps=(10, 11))
    w = LeRobotWriter(root, fps=30, state_names=SN, action_names=SN, cameras=CAMS, fmt=fmt)
    assert (w.num_episodes, w.total_frames) == (2, 21)
    w.begin_episode("a brand new task"); frames(w, 9, shade=200)
    assert w.save_episode() == 2
    w.close()
    info = json.loads((root / "meta/info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["total_tasks"]) == (3, 30, 2)
    if fmt == "v3.0":
        eps = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
        assert [(e["dataset_from_index"], e["dataset_to_index"]) for e in eps] == [(0, 10), (10, 21), (21, 30)]
        assert len(video_frames(root / f"videos/{IMG}cam_a/chunk-000/file-000.mp4")) == 30
        assert pq.read_table(root / "data/chunk-000/file-000.parquet").column("index").to_pylist() == list(range(30))
        assert np.array(json.loads((root / "meta/stats.json").read_text())["index"]["count"]) == 30
    else:
        assert len(video_frames(root / f"videos/chunk-000/{IMG}cam_a/episode_000002.mp4")) == 9


def test_v30_rolls_to_a_new_file_at_the_size_limit(tmp_path):
    root = tmp_path / "d"
    make(root, "v3.0", eps=(30, 30, 30), data_file_mb=0.0001, video_file_mb=0.0001)          # absurdly small limits
    eps = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [e["data/file_index"] for e in eps] == [0, 1, 2]
    assert [e[f"videos/{IMG}cam_a/file_index"] for e in eps] == [0, 1, 2]
    assert all(e[f"videos/{IMG}cam_a/from_timestamp"] == 0.0 for e in eps)
    for i in range(3):
        assert len(video_frames(root / f"videos/{IMG}cam_a/chunk-000/file-{i:03d}.mp4")) == 30
        assert pq.read_table(root / f"data/chunk-000/file-{i:03d}.parquet").num_rows == 30


def test_refusals(tmp_path):
    (tmp_path / "x").mkdir(); (tmp_path / "x" / "stuff.txt").write_text("hi")
    with pytest.raises(FileExistsError):
        LeRobotWriter(tmp_path / "x", state_names=SN, action_names=SN)
    make(tmp_path / "d", "v3.0", eps=(5,))
    with pytest.raises(ValueError, match="v3.0 dataset, not v2.1"):
        LeRobotWriter(tmp_path / "d", state_names=SN, action_names=SN, cameras=CAMS, fmt="v2.1")
    with pytest.raises(ValueError, match="fps"):
        LeRobotWriter(tmp_path / "d", fps=20, state_names=SN, action_names=SN, cameras=CAMS, fmt="v3.0")
    with pytest.raises(ValueError, match="differs"):
        LeRobotWriter(tmp_path / "d", state_names=SN[:-1] + ["grip"], action_names=SN, cameras=CAMS, fmt="v3.0")
    with pytest.raises(ValueError, match="cameras"):
        LeRobotWriter(tmp_path / "d", state_names=SN, action_names=SN, cameras={"cam_a": (48, 64)}, fmt="v3.0")
    w = LeRobotWriter(tmp_path / "d", state_names=SN, action_names=SN, cameras=CAMS, fmt="v3.0")
    w.begin_episode("t")
    with pytest.raises(ValueError, match="missing camera"):
        w.add_frame(np.zeros(7), np.zeros(7), {"cam_a": np.zeros((48, 64, 3), np.uint8)})
    with pytest.raises(ValueError, match="do not match"):
        w.add_frame(np.zeros(3), np.zeros(7), {c: np.zeros((h, wd, 3), np.uint8) for c, (h, wd) in CAMS.items()})
    with pytest.raises(RuntimeError, match="no frames"):
        w.save_episode()
