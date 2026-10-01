"""Timeline visualisation."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _colors(labels):
    cmap = plt.get_cmap("tab20")
    return {l: cmap(i % 20) for i, l in enumerate(sorted(set(labels)))}


def plot_timeline(segments, duration, gt_segments=None, title="Detected activity timeline"):
    segs = [s if isinstance(s, dict) else s.to_dict() for s in segments]
    gts = [s if isinstance(s, dict) else s.to_dict() for s in (gt_segments or [])]
    col = _colors([s["label"] for s in segs + gts])

    rows = (["gt"] if gts else []) + ["pred", "conf"]
    ratios = {"gt": 1, "pred": 1, "conf": 1.6}
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.2 * len(rows) + 1.6), sharex=True,
                             gridspec_kw={"height_ratios": [ratios[r] for r in rows]})
    axes = np.atleast_1d(axes)

    def draw_bars(ax, items, name):
        for s in items:
            w = s["end"] - s["start"]
            ax.barh(0, w, left=s["start"], color=col[s["label"]], edgecolor="white", height=0.9)
            if w / max(duration, 1e-9) > 0.07:
                ax.text(s["start"] + w / 2, 0, s["label"][:20], ha="center", va="center", fontsize=8)
        ax.set_yticks([]); ax.set_ylabel(name, rotation=0, ha="right", va="center")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    i = 0
    if gts:
        draw_bars(axes[i], gts, "ground truth"); i += 1
    draw_bars(axes[i], segs, "predicted"); i += 1

    ax = axes[i]
    for s in segs:
        ax.bar(s["start"], s["confidence"], width=s["end"] - s["start"], align="edge",
               color=col[s["label"]], edgecolor="white")
    ax.set_ylim(0, 1); ax.set_ylabel("confidence", rotation=0, ha="right", va="center")
    ax.set_xlabel("time (s)"); ax.set_xlim(0, duration)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    axes[0].set_title(title)
    fig.tight_layout()
    return fig
