"""Write LeRobot datasets (codebase v2.1 or v3.0) without lerobot / torch.

    w = LeRobotWriter("datasets/pick_cube", fps=30, state_names=[...], action_names=[...],
                      cameras={"wrist": (480, 640)}, fmt="v3.0")        # opens an existing dataset too (appends)
    w.begin_episode("pick up the cube")
    w.add_frame(state, action, {"wrist": rgb_uint8_hxwx3})              # 30x per second
    w.save_episode()      # or w.discard_episode()
    w.close()

Layouts (checked against datasets written by LeRobot itself):

v3.0   meta/info.json, meta/stats.json, meta/tasks.parquet, meta/episodes/chunk-000/file-000.parquet,
       data/chunk-000/file-000.parquet            all episodes, rolled to a new file at `data_file_mb`
       videos/<key>/chunk-000/file-000.mp4        all episodes, rolled at `video_file_mb`; an episode is found by
                                                  from_timestamp / to_timestamp in its episodes row
v2.1   meta/info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl
       data/chunk-000/episode_000000.parquet, videos/chunk-000/<key>/episode_000000.mp4

Everything an episode needs is written before the metadata that references it, so an interrupted save leaves a dataset
that still loads (the unfinished episode is simply absent).  Episodes in progress live in `<root>/.inflight/`.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .stats import ImageStatsAccumulator, aggregate, to_jsonable, vector_stats
from .video import StreamEncoder, concat_mp4

IMG = "observation.images."
CODEC_NAME = {"libx264": "h264", "h264": "h264", "libx265": "hevc", "libaom-av1": "av1", "libsvtav1": "av1"}
BOOKKEEPING = ("timestamp", "frame_index", "episode_index", "index", "task_index")
STAT_KEYS_V21 = ("min", "max", "mean", "std", "count")


def _thumb(rgb: np.ndarray, side: int = 96) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max(1, int(np.ceil(max(h, w) / side)))
    try:
        import cv2
        return cv2.resize(rgb, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
    except ImportError:
        return rgb[::s, ::s]


def _hf_feature(name: str, spec: dict) -> dict:
    if spec["dtype"] in ("float32", "int64"):
        d = spec["shape"][0]
        if name in ("action", "observation.state"):
            return {"feature": {"dtype": spec["dtype"], "_type": "Value"}, "length": d, "_type": "List"}
        return {"dtype": spec["dtype"], "_type": "Value"}
    raise ValueError(name)


class LeRobotWriter:
    def __init__(self, root, *, fps: int = 30, state_names, action_names, cameras: dict | None = None,
                 robot_type: str = "piper", fmt: str = "v3.0", vcodec: str = "libx264", crf: int = 23, gop: int = 2,
                 data_file_mb: float = 100, video_file_mb: float = 200, chunks_size: int = 1000, image_stat_stride: int = 3):
        if fmt not in ("v2.1", "v3.0"):
            raise ValueError("fmt must be 'v2.1' or 'v3.0'")
        self.root = Path(root)
        self.fmt, self.fps = fmt, int(fps)
        self.state_names, self.action_names = list(state_names), list(action_names)
        self.cameras = dict(cameras or {})                       # name -> (height, width)
        self.robot_type = robot_type
        self.vcodec, self.crf, self.gop = vcodec, crf, gop
        self.data_limit, self.video_limit = data_file_mb * 1e6, video_file_mb * 1e6
        self.chunks_size = chunks_size
        self.stat_stride = max(1, image_stat_stride)
        self.features = self._build_features()
        self.tasks: dict[str, int] = {}
        self.episode_rows: list[dict] = []                       # v3: the episodes table; v2.1: episodes.jsonl rows
        self.episode_stats: list[dict] = []                      # per episode: feature -> stats dict (numpy)
        self.total_frames = 0
        self._ep = None
        self._data_ptr = (0, 0)                                  # v3: (chunk, file) being appended to
        self._vid_ptr: dict[str, tuple[int, int, int]] = {}      # v3: camera -> (chunk, file, frames in that file)
        self._open()

    # ----------------------------------------------------------------- setup
    def _build_features(self) -> dict:
        f = {"action": {"dtype": "float32", "shape": [len(self.action_names)], "names": self.action_names},
             "observation.state": {"dtype": "float32", "shape": [len(self.state_names)], "names": self.state_names}}
        for cam, (h, w) in self.cameras.items():
            f[IMG + cam] = {"dtype": "video", "shape": [int(h), int(w), 3], "names": ["height", "width", "channels"],
                            "info": {"video.height": int(h), "video.width": int(w),
                                     "video.codec": CODEC_NAME.get(self.vcodec, self.vcodec), "video.pix_fmt": "yuv420p",
                                     "video.is_depth_map": False, "video.fps": self.fps, "video.channels": 3,
                                     "has_audio": False}}
        for k, dt in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"),
                      ("index", "int64"), ("task_index", "int64")):
            f[k] = {"dtype": dt, "shape": [1], "names": None}
        return f

    def _open(self):
        info_p = self.root / "meta" / "info.json"
        if info_p.exists():
            self._resume(json.loads(info_p.read_text()))
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise FileExistsError(f"{self.root} exists and is not a dataset - choose an empty directory")
            (self.root / "meta").mkdir(parents=True, exist_ok=True)
            self._write_info()
        shutil.rmtree(self.root / ".inflight", ignore_errors=True)

    def _resume(self, info: dict):
        if info.get("codebase_version") != self.fmt:
            raise ValueError(f"{self.root} is a {info.get('codebase_version')} dataset, not {self.fmt} "
                             "(pick the same --format, or another directory)")
        if info["fps"] != self.fps:
            raise ValueError(f"{self.root} was recorded at {info['fps']} fps, not {self.fps}")
        for k in ("action", "observation.state") + tuple(IMG + c for c in self.cameras):
            old, new = info["features"].get(k), self.features[k]
            if old is None or old["shape"] != new["shape"] or old.get("names") != new.get("names"):
                raise ValueError(f"feature '{k}' differs from the existing dataset ({old} vs {new}); cannot append")
        extra = [k for k in info["features"] if k.startswith(IMG) and k[len(IMG):] not in self.cameras]
        if extra:
            raise ValueError(f"the existing dataset has cameras {extra} that this session does not provide")
        self.total_frames = int(info["total_frames"])
        if self.fmt == "v2.1":
            self.tasks = {r["task"]: r["task_index"] for r in self._read_jsonl("tasks.jsonl")}
            self.episode_rows = self._read_jsonl("episodes.jsonl")
            self.episode_stats = [{k: {a: np.asarray(b) for a, b in v.items()} for k, v in r["stats"].items()}
                                  for r in self._read_jsonl("episodes_stats.jsonl")]
        else:
            t = pq.read_table(self.root / "meta" / "tasks.parquet").to_pylist()
            self.tasks = {r["task"]: r["task_index"] for r in t}
            self.episode_rows = pq.read_table(self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
            self.episode_stats = [self._stats_from_row(r) for r in self.episode_rows]
            if self.episode_rows:
                last = self.episode_rows[-1]
                self._data_ptr = (last["data/chunk_index"], last["data/file_index"])
                for cam in self.cameras:
                    k = f"videos/{IMG}{cam}/"
                    self._vid_ptr[cam] = (last[k + "chunk_index"], last[k + "file_index"],
                                          int(round(last[k + "to_timestamp"] * self.fps)))
        if len(self.episode_rows) != int(info["total_episodes"]):
            raise ValueError(f"{self.root}: info.json says {info['total_episodes']} episodes but the episode table has "
                             f"{len(self.episode_rows)} - the dataset is inconsistent")

    def _read_jsonl(self, name: str) -> list[dict]:
        p = self.root / "meta" / name
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    @staticmethod
    def _stats_from_row(row: dict) -> dict:
        out: dict = {}
        for k, v in row.items():
            if k.startswith("stats/"):
                feat, stat = k[len("stats/"):].rsplit("/", 1)
                out.setdefault(feat, {})[stat] = np.asarray(v, dtype=np.int64 if stat == "count" else np.float64)
        return out

    # ------------------------------------------------------------- properties
    @property
    def num_episodes(self) -> int:
        return len(self.episode_rows)

    @property
    def in_episode(self) -> bool:
        return self._ep is not None

    @property
    def frames_in_episode(self) -> int:
        return len(self._ep["state"]) if self._ep else 0

    # -------------------------------------------------------------- recording
    def begin_episode(self, task: str):
        if self._ep is not None:
            raise RuntimeError("an episode is already open - save or discard it first")
        idx = self.num_episodes
        tmp = self.root / ".inflight" / f"ep{idx:06d}"
        shutil.rmtree(tmp, ignore_errors=True)
        enc = {c: StreamEncoder(tmp / f"{c}.mp4", w, h, self.fps, self.vcodec, self.crf, self.gop)
               for c, (h, w) in self.cameras.items()}
        self._ep = {"index": idx, "task": task, "tmp": tmp, "enc": enc, "state": [], "action": [],
                    "img_stats": {c: ImageStatsAccumulator() for c in self.cameras}}

    def add_frame(self, state, action, images: dict | None = None):
        ep = self._ep
        if ep is None:
            raise RuntimeError("begin_episode() first")
        state = np.asarray(state, dtype=np.float32).ravel()
        action = np.asarray(action, dtype=np.float32).ravel()
        if state.shape != (len(self.state_names),) or action.shape != (len(self.action_names),):
            raise ValueError(f"state {state.shape} / action {action.shape} do not match the declared "
                             f"{len(self.state_names)} / {len(self.action_names)} values")
        images = images or {}
        missing = [c for c in self.cameras if c not in images]
        if missing:
            raise ValueError(f"missing camera frame(s): {missing}")
        n = len(ep["state"])
        for c, enc in ep["enc"].items():
            enc.put(images[c])
            if n % self.stat_stride == 0:
                ep["img_stats"][c].add(_thumb(np.asarray(images[c])))
        ep["state"].append(state)
        ep["action"].append(action)

    def discard_episode(self):
        ep, self._ep = self._ep, None
        if ep is None:
            return
        for enc in ep["enc"].values():
            try:
                enc.close()
            except Exception:
                pass
        shutil.rmtree(ep["tmp"], ignore_errors=True)

    def save_episode(self) -> int:
        ep = self._ep
        if ep is None or not ep["state"]:
            raise RuntimeError("no frames in this episode - discard it instead")
        self._ep = None
        try:
            for c, enc in ep["enc"].items():
                if enc.close() != len(ep["state"]):
                    raise RuntimeError(f"camera '{c}': {enc.n} video frames for {len(ep['state'])} samples")
            return self._commit(ep)
        finally:
            shutil.rmtree(ep["tmp"], ignore_errors=True)

    # ------------------------------------------------------------------ commit
    def _task_index(self, task: str) -> int:
        if task not in self.tasks:
            self.tasks[task] = len(self.tasks)
        return self.tasks[task]

    def _episode_stats(self, ep: dict, table_cols: dict) -> dict:
        st = {"action": vector_stats(np.stack(ep["action"])), "observation.state": vector_stats(np.stack(ep["state"]))}
        for k in BOOKKEEPING:
            st[k] = vector_stats(table_cols[k])
        for c in self.cameras:
            st[IMG + c] = ep["img_stats"][c].result()
        return st

    def _table(self, ep: dict, start: int, task_index: int):
        n = len(ep["state"])
        cols = {"timestamp": (np.arange(n) / self.fps).astype(np.float32), "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, ep["index"], np.int64), "index": np.arange(start, start + n, dtype=np.int64),
                "task_index": np.full(n, task_index, np.int64)}

        def vec(a):
            flat = pa.array(np.concatenate(a).astype(np.float32))
            d = len(a[0])
            if self.fmt == "v3.0":
                return pa.FixedSizeListArray.from_arrays(flat, d)
            return pa.ListArray.from_arrays(pa.array(np.arange(0, (n + 1) * d, d, dtype=np.int32)), flat)

        arrays = {"action": vec(ep["action"]), "observation.state": vec(ep["state"])}
        arrays.update({k: pa.array(v) for k, v in cols.items()})
        table = pa.table(arrays)
        if self.fmt == "v3.0":                                     # the 'huggingface' block datasets/LeRobot expect
            feats = {k: _hf_feature(k, self.features[k]) for k in table.column_names}
            table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": feats}})})
        return table, cols

    def _commit(self, ep: dict) -> int:
        idx, n = ep["index"], len(ep["state"])
        ti = self._task_index(ep["task"])
        table, cols = self._table(ep, self.total_frames, ti)
        stats = self._episode_stats(ep, cols)
        if self.fmt == "v2.1":
            row = self._commit_v21(ep, table)
        else:
            row = self._commit_v30(ep, table)
        row.update({"episode_index": idx, "tasks": [ep["task"]], "length": n})
        self.episode_rows.append(row)
        self.episode_stats.append(stats)
        self.total_frames += n
        self._write_meta()
        return idx

    def _commit_v21(self, ep: dict, table) -> dict:
        idx, chunk = ep["index"], ep["index"] // self.chunks_size
        dst = self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{idx:06d}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst)
        for c in self.cameras:
            vdst = self.root / "videos" / f"chunk-{chunk:03d}" / f"{IMG}{c}" / f"episode_{idx:06d}.mp4"
            vdst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(ep["tmp"] / f"{c}.mp4", vdst)
        return {}

    def _commit_v30(self, ep: dict, table) -> dict:
        n = len(ep["state"])
        row: dict = {}
        # data: append to the current file unless it would pass the size limit
        chunk, file = self._data_ptr
        path = self._data_path(chunk, file)
        if path.exists() and path.stat().st_size + table.nbytes > self.data_limit:
            chunk, file = self._next_file(chunk, file)
            path = self._data_path(chunk, file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            table = pa.concat_tables([pq.read_table(path), table])
        tmp = path.with_suffix(".tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, path)
        self._data_ptr = (chunk, file)
        row.update({"data/chunk_index": chunk, "data/file_index": file,
                    "dataset_from_index": self.total_frames, "dataset_to_index": self.total_frames + n})
        # videos: same, per camera, by concatenating packets
        for c in self.cameras:
            vchunk, vfile, vframes = self._vid_ptr.get(c, (0, 0, 0))
            src = ep["tmp"] / f"{c}.mp4"
            vpath = self._video_path(c, vchunk, vfile)
            if vpath.exists() and vpath.stat().st_size + src.stat().st_size > self.video_limit:
                vchunk, vfile = self._next_file(vchunk, vfile)
                vpath, vframes = self._video_path(c, vchunk, vfile), 0
            vpath.parent.mkdir(parents=True, exist_ok=True)
            if vpath.exists():
                tmpv = vpath.with_suffix(".tmp.mp4")
                concat_mp4(tmpv, [vpath, src])
                os.replace(tmpv, vpath)
            else:
                os.replace(src, vpath)
            k = f"videos/{IMG}{c}/"
            row.update({k + "chunk_index": vchunk, k + "file_index": vfile,
                        k + "from_timestamp": vframes / self.fps, k + "to_timestamp": (vframes + n) / self.fps})
            self._vid_ptr[c] = (vchunk, vfile, vframes + n)
        row.update({"meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
        return row

    def _next_file(self, chunk: int, file: int):
        return (chunk + 1, 0) if file + 1 >= self.chunks_size else (chunk, file + 1)

    def _data_path(self, chunk, file) -> Path:
        return self.root / "data" / f"chunk-{chunk:03d}" / f"file-{file:03d}.parquet"

    def _video_path(self, cam, chunk, file) -> Path:
        return self.root / "videos" / f"{IMG}{cam}" / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"

    # ---------------------------------------------------------------- metadata
    def _write_info(self):
        n_ep = self.num_episodes
        info = {"codebase_version": self.fmt, "robot_type": self.robot_type, "total_episodes": n_ep,
                "total_frames": self.total_frames, "total_tasks": len(self.tasks), "chunks_size": self.chunks_size}
        if self.fmt == "v3.0":
            info.update({"data_files_size_in_mb": int(self.data_limit / 1e6), "video_files_size_in_mb": int(self.video_limit / 1e6)})
        else:
            info.update({"total_videos": n_ep * len(self.cameras), "total_chunks": (n_ep + self.chunks_size - 1) // self.chunks_size})
        info.update({"fps": self.fps, "splits": {"train": f"0:{n_ep}"}})
        if self.fmt == "v3.0":
            info.update({"data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                         "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"})
        else:
            info.update({"data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                         "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"})
        info["features"] = self.features
        self._atomic_text(self.root / "meta" / "info.json", json.dumps(info, indent=4))

    @staticmethod
    def _atomic_text(path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)

    def _write_meta(self):
        meta = self.root / "meta"
        tasks = sorted(self.tasks.items(), key=lambda kv: kv[1])
        if self.fmt == "v2.1":
            self._atomic_text(meta / "tasks.jsonl", "".join(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tasks))
            self._atomic_text(meta / "episodes.jsonl", "".join(json.dumps(r) + "\n" for r in self.episode_rows))
            self._atomic_text(meta / "episodes_stats.jsonl", "".join(
                json.dumps({"episode_index": i, "stats": {f: {k: np.asarray(v).tolist() for k, v in s.items() if k in STAT_KEYS_V21}
                                                              for f, s in st.items()}}) + "\n"
                for i, st in enumerate(self.episode_stats)))
        else:
            self._write_tasks_parquet(tasks)
            rows = []
            for r, st in zip(self.episode_rows, self.episode_stats):
                full = dict(r)
                for feat, s in st.items():
                    for k, v in s.items():
                        full[f"stats/{feat}/{k}"] = np.asarray(v).tolist()
                rows.append(full)
            t = pa.Table.from_pylist(rows)
            p = meta / "episodes" / "chunk-000" / "file-000.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, p.with_suffix(".tmp"))
            os.replace(p.with_suffix(".tmp"), p)
            agg = {feat: aggregate([st[feat] for st in self.episode_stats]) for feat in self.episode_stats[0]}
            self._atomic_text(meta / "stats.json", json.dumps({f: to_jsonable(s) for f, s in agg.items()}, indent=4))
        self._write_info()

    def _write_tasks_parquet(self, tasks):
        # LeRobot reads this with pandas (index = task text, one column task_index), so mark `task` as the index
        schema = pa.schema([pa.field("task_index", pa.int64()), pa.field("task", pa.large_string())])
        pandas_meta = {"index_columns": ["task"], "column_indexes": [{"name": None, "field_name": None, "pandas_type": "unicode",
                       "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}],
                       "columns": [{"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
                                    "numpy_type": "int64", "metadata": None},
                                   {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object",
                                    "metadata": None}],
                       "creator": {"library": "pyarrow", "version": pa.__version__}, "pandas_version": "2.2.0"}
        t = pa.table({"task_index": [i for _, i in tasks], "task": [t for t, _ in tasks]}, schema=schema)
        t = t.replace_schema_metadata({"pandas": json.dumps(pandas_meta)})
        p = self.root / "meta" / "tasks.parquet"
        pq.write_table(t, p.with_suffix(".tmp"))
        os.replace(p.with_suffix(".tmp"), p)

    def log_session(self, info: dict):
        """Append a record of how a session was recorded (arms, mapping, camera settings ...) to meta/recording_setup.json.
        LeRobot ignores the file; it is there so a policy can later be fed images that look like the training ones."""
        p = self.root / "meta" / "recording_setup.json"
        data = json.loads(p.read_text()) if p.exists() else {"sessions": []}
        data["sessions"].append(info)
        self._atomic_text(p, json.dumps(data, indent=2, default=str))

    def close(self):
        self.discard_episode()
