"""Turn per-window predictions into clean time segments (no training involved).

Two methods:
  merge       : smooth class probabilities, take argmax per time bin, merge runs,
                absorb too-short segments into a neighbour.
  changepoint : detect boundaries from embedding (or probability) changes with PELT,
                then label each segment by averaging class probabilities inside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass
class Segment:
    start: float
    end: float
    label: str
    confidence: float
    alternatives: list = field(default_factory=list)   # [(label, prob), ...]

    def to_dict(self) -> dict:
        return {
            "start": round(float(self.start), 2),
            "end": round(float(self.end), 2),
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "alternatives": [{"label": l, "prob": round(float(p), 4)}
                             for l, p in self.alternatives],
        }


# ---------------------------------------------------------------- helpers
def windows_to_grid(values: np.ndarray, win_start: np.ndarray, win_end: np.ndarray,
                    duration: float, step: float) -> np.ndarray:
    """Average window-level vectors onto a uniform time grid (bin width = step)."""
    n_bins = max(1, int(np.ceil(duration / step - 1e-9)))
    centers = (np.arange(n_bins) + 0.5) * step
    out = np.zeros((n_bins, values.shape[1]), dtype=np.float64)
    cnt = np.zeros(n_bins)
    for v, s, e in zip(values, win_start, win_end):
        lo = int(np.searchsorted(centers, s, side="left"))
        hi = int(np.searchsorted(centers, e, side="left"))
        if hi <= lo:
            lo = min(max(int(((s + e) / 2) / step), 0), n_bins - 1)
            hi = lo + 1
        out[lo:hi] += v
        cnt[lo:hi] += 1
    missing = cnt == 0
    if missing.any():
        valid = np.where(~missing)[0]
        for m in np.where(missing)[0]:
            j = valid[np.argmin(np.abs(valid - m))]
            out[m], cnt[m] = out[j] * cnt[j], cnt[j]
    return out / np.maximum(cnt, 1)[:, None]


def _runs(lab: np.ndarray):
    runs, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            runs.append((int(lab[s]), s, i))
            s = i
    return runs


def _absorb_short(lab: np.ndarray, P: np.ndarray, min_bins: int) -> np.ndarray:
    lab = lab.copy()
    while True:
        runs = _runs(lab)
        short = [(i, r) for i, r in enumerate(runs) if r[2] - r[1] < min_bins]
        if len(runs) <= 1 or not short:
            return lab
        i, (_, s, e) = min(short, key=lambda x: x[1][2] - x[1][1])
        cands = []
        if i > 0:
            cands.append(runs[i - 1][0])
        if i < len(runs) - 1:
            cands.append(runs[i + 1][0])
        lab[s:e] = max(cands, key=lambda c: P[s:e, c].mean())


def _build(bin_labels, P, step, duration, names, top_k=3) -> list[Segment]:
    segs = []
    for lab, s, e in _runs(bin_labels):
        m = P[s:e].mean(axis=0)
        order = np.argsort(-m)
        alts = [(names[j], float(m[j])) for j in order if j != lab][:top_k - 1]
        segs.append(Segment(s * step, min(e * step, duration), names[lab],
                            float(m[lab]), alts))
    return segs


# ---------------------------------------------------------------- methods
def segment_merge(P, step, duration, names, smooth_sec=2.0, min_dur=2.0):
    k = max(1, int(round(smooth_sec / step)))
    Ps = uniform_filter1d(P, size=k, axis=0, mode="nearest") if k > 1 else P
    lab = _absorb_short(Ps.argmax(axis=1), Ps, max(1, int(round(min_dur / step))))
    return _build(lab, Ps, step, duration, names)


def segment_changepoint(P, E, step, duration, names, min_dur=2.0, penalty_scale=4.0,
                        feature="embed", smooth_sec=1.0, pca_dims=16):
    import ruptures as rpt

    n = len(P)
    min_bins = max(1, int(round(min_dur / step)))
    one = lambda: _build(np.full(n, int(P.mean(0).argmax())), P, step, duration, names)
    if n < 2 * min_bins:
        return one()

    X = np.sqrt(P) if feature == "probs" else E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    k = max(1, int(round(smooth_sec / step)))
    if k > 1:
        X = uniform_filter1d(X, size=k, axis=0, mode="nearest")

    Xc = X - X.mean(axis=0)                      # PCA: denoise + keep PELT fast
    d = min(pca_dims, n, X.shape[1])
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ vt[:d].T

    total_var = float(Z.var(axis=0).sum())
    if total_var < 1e-10:
        return one()
    pen = penalty_scale * total_var * np.log(n)  # scale-free: relative to sequence variance
    bkps = rpt.Pelt(model="l2", min_size=min_bins, jump=1).fit(Z).predict(pen=pen)

    lab, prev = np.zeros(n, dtype=int), 0
    for b in bkps:
        lab[prev:b] = int(P[prev:b].mean(axis=0).argmax())
        prev = b
    return _build(lab, P, step, duration, names)   # adjacent equal labels merge here


def segment(probs, embeds, win_start, win_end, duration, names, method="changepoint",
            step=0.5, smooth_sec=2.0, min_dur=2.0, penalty_scale=4.0, feature="embed"):
    P = windows_to_grid(probs, win_start, win_end, duration, step)
    if method == "merge":
        return segment_merge(P, step, duration, names, smooth_sec, min_dur)
    if method == "changepoint":
        E = windows_to_grid(embeds, win_start, win_end, duration, step)
        return segment_changepoint(P, E, step, duration, names, min_dur, penalty_scale,
                                   feature, smooth_sec=min(smooth_sec, 1.0))
    raise ValueError(f"Unknown method: {method}")
