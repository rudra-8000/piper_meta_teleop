"""Dataset statistics in the layout LeRobot writes (min / max / mean / std / count / q01 q10 q50 q90 q99).

Vector features are computed exactly from the episode's arrays.  Image features are computed from a sparse, downsampled
set of frames through a 256-bin histogram per channel (exact for that set, O(1) memory): values normalised to [0, 1],
shape (3, 1, 1), like LeRobot.
"""
from __future__ import annotations

import numpy as np

QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def vector_stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    out = {"min": a.min(0), "max": a.max(0), "mean": a.mean(0), "std": a.std(0), "count": np.array([len(a)])}
    for k, q in QUANTILES.items():
        out[k] = np.quantile(a, q, axis=0)
    return out


class ImageStatsAccumulator:
    """Per-channel histogram of uint8 pixels from a sparse set of frames."""

    def __init__(self):
        self.hist = np.zeros((3, 256), dtype=np.int64)
        self.n_frames = 0

    def add(self, frame_rgb: np.ndarray):
        f = np.asarray(frame_rgb)
        for c in range(3):
            self.hist[c] += np.bincount(f[..., c].ravel(), minlength=256)
        self.n_frames += 1

    def result(self) -> dict:
        h = self.hist.astype(np.float64)
        n = h.sum(1)                                     # pixels per channel
        v = np.arange(256, dtype=np.float64)
        mean = (h * v).sum(1) / n
        std = np.sqrt(np.maximum((h * v * v).sum(1) / n - mean ** 2, 0.0))
        nz = [np.nonzero(h[c])[0] for c in range(3)]
        out = {"min": np.array([z[0] for z in nz], float), "max": np.array([z[-1] for z in nz], float), "mean": mean, "std": std}
        cdf = np.cumsum(h, 1) / n[:, None]
        for k, q in QUANTILES.items():
            out[k] = np.array([float(np.searchsorted(cdf[c], q)) for c in range(3)])
        out = {k: (v_ / 255.0).reshape(3, 1, 1) for k, v_ in out.items()}
        out["count"] = np.array([self.n_frames])
        return out


def aggregate(stats_list: list[dict]) -> dict:
    """Combine per-episode stats of ONE feature: min/max extremes, count-weighted mean, pooled std, weighted quantiles."""
    counts = np.array([float(np.ravel(s["count"])[0]) for s in stats_list])
    w = counts / counts.sum()
    shape = np.asarray(stats_list[0]["mean"]).shape

    def stack(k):
        return np.stack([np.asarray(s[k], float) for s in stats_list])

    wb = w.reshape((-1,) + (1,) * len(shape))
    mean = (stack("mean") * wb).sum(0)
    var = (wb * (stack("std") ** 2 + (stack("mean") - mean) ** 2)).sum(0)
    out = {"min": stack("min").min(0), "max": stack("max").max(0), "mean": mean, "std": np.sqrt(var),
           "count": np.array([int(counts.sum())])}
    for k in QUANTILES:
        if all(k in s for s in stats_list):
            out[k] = (stack(k) * wb).sum(0)
    return out


def to_jsonable(stats: dict) -> dict:
    return {k: np.asarray(v).tolist() for k, v in stats.items()}
