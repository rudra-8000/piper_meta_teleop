#!/usr/bin/env python3
"""Check a recorded LeRobot dataset (v2.1 or v3.0) before you push it or train on it.

    .venv/bin/python tools/validate_dataset.py datasets/pick_cube [--lerobot]

Structure checks need only pyarrow + av: info.json against the files, every episode's rows / video frames / timestamps,
contiguous indices, task table.  With --lerobot (needs lerobot installed - e.g. on your training machine) it also loads the
dataset with LeRobotDataset and decodes one frame from every camera of the first and last episode.
Exit code 1 if anything is wrong.
"""
import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--lerobot", action="store_true", help="also load it with the real LeRobotDataset class")
a = ap.parse_args()
root = Path(a.root)
bad: list[str] = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        bad.append(msg)


def n_frames(path):
    with av.open(str(path)) as c:
        return sum(1 for _ in c.decode(video=0))


info = json.loads((root / "meta/info.json").read_text())
ver, fps = info["codebase_version"], info["fps"]
cams = [k for k, v in info["features"].items() if v["dtype"] == "video"]
print(f"{root}: {ver}, {info['total_episodes']} episodes, {info['total_frames']} frames, {fps} fps, cameras {cams or 'none'}")
check(ver in ("v2.1", "v3.0"), f"codebase_version {ver} is v2.1 or v3.0")

if ver == "v3.0":
    eps = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    table = pa.concat_tables([pq.read_table(f) for f in files])
    check(len(eps) == info["total_episodes"], f"{len(eps)} rows in the episodes table == total_episodes")
    check(table.num_rows == info["total_frames"], f"{table.num_rows} data rows == total_frames")
    idx = table.column("index").to_numpy()
    check(np.array_equal(idx, np.arange(len(idx))), "frame index runs 0..N-1 without gaps")
    ep_col = table.column("episode_index").to_numpy()
    for e in eps:
        i, lo, hi = e["episode_index"], e["dataset_from_index"], e["dataset_to_index"]
        check(hi - lo == e["length"] and set(ep_col[lo:hi]) == {i}, f"episode {i}: rows {lo}..{hi} belong to it ({e['length']} frames)")
        ts = table.column("timestamp").to_numpy()[lo:hi]
        check(np.allclose(ts, np.arange(e["length"]) / fps, atol=1e-4), f"episode {i}: timestamps are frame/{fps}")
        for c in cams:
            k = f"videos/{c}/"
            check(abs((e[k + "to_timestamp"] - e[k + "from_timestamp"]) * fps - e["length"]) < 1e-6, f"episode {i} {c}: video span == {e['length']} frames")
    for c in cams:
        vids = sorted((root / "videos" / c).glob("chunk-*/file-*.mp4"))
        total = sum(n_frames(v) for v in vids)
        check(total == info["total_frames"], f"{c}: {len(vids)} video file(s) hold {total} frames == total_frames")
    check(set(pq.read_table(root / "meta/tasks.parquet").column("task_index").to_pylist()) ==
          {t for e in eps for t in [pq.read_table(files[0]).column("task_index")[0].as_py()]} | set(table.column("task_index").to_pylist()),
          "task_index values are all in tasks.parquet")
    check((root / "meta/stats.json").exists(), "meta/stats.json exists")
else:
    eps = [json.loads(l) for l in (root / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    check(len(eps) == info["total_episodes"], f"{len(eps)} lines in episodes.jsonl == total_episodes")
    total = 0
    for e in eps:
        i, n = e["episode_index"], e["length"]
        t = pq.read_table(root / f"data/chunk-{i // info['chunks_size']:03d}/episode_{i:06d}.parquet")
        check(t.num_rows == n, f"episode {i}: {t.num_rows} rows == length {n}")
        check(set(t.column("episode_index").to_pylist()) == {i} and np.allclose(t.column("timestamp").to_numpy(), np.arange(n) / fps, atol=1e-4),
              f"episode {i}: episode_index and timestamps")
        total += n
        for c in cams:
            vf = n_frames(root / f"videos/chunk-{i // info['chunks_size']:03d}/{c}/episode_{i:06d}.mp4")
            check(vf == n, f"episode {i} {c}: {vf} video frames == {n}")
    check(total == info["total_frames"], f"episode lengths add up to {total} == total_frames")
    check((root / "meta/episodes_stats.jsonl").exists(), "meta/episodes_stats.jsonl exists")

if a.lerobot:
    try:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset       # older / OpenPI layout
        ds = LeRobotDataset(root.name, root=str(root))
        check(len(ds) == info["total_frames"], f"LeRobotDataset loads: {len(ds)} frames, {ds.num_episodes} episodes")
        for idx in (0, len(ds) - 1):
            item = ds[idx]
            for c in cams:
                check(tuple(item[c].shape)[0] == 3, f"frame {idx} {c} decodes to {tuple(item[c].shape)}")
    except Exception as e:                                                          # noqa: BLE001
        check(False, f"LeRobotDataset: {type(e).__name__}: {str(e)[:160]}")

print("\nOK" if not bad else f"\n{len(bad)} problem(s)")
sys.exit(1 if bad else 0)
