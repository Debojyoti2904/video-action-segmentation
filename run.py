"""Command-line interface.

    python run.py path/to/video.mp4
    python run.py video.mp4 --labels "chopping vegetables, stirring, plating"
    python run.py --list-labels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.model import DEFAULT_XCLIP_LABELS, load_model
from src.pipeline import analyze, segment_analysis
from src.viz import plot_timeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--labels", default=None, help="comma-separated labels")
    ap.add_argument("--method", choices=["merge", "changepoint"], default="changepoint")
    ap.add_argument("--feature", choices=["embed", "probs"], default="embed")
    ap.add_argument("--smooth", type=float, default=2.0, help="seconds (merge method)")
    ap.add_argument("--min-dur", type=float, default=2.0, help="min segment length, seconds")
    ap.add_argument("--penalty", type=float, default=4.0, help="higher = fewer boundaries")
    ap.add_argument("--hop", type=int, default=4, help="window hop in frames @ 8 fps (4 = 0.5 s)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--list-labels", action="store_true")
    a = ap.parse_args()

    labels = [l.strip() for l in a.labels.split(",") if l.strip()] if a.labels else DEFAULT_XCLIP_LABELS
    model = load_model(labels=labels)
    if a.list_labels:
        print("\n".join(model.labels))
        return
    if not a.video:
        ap.error("video path required")

    an = analyze(a.video, model, hop_frames=a.hop,
                 progress_cb=lambda p: print(f"\rinference {p:6.1%}", end="", flush=True))
    print()
    segs = segment_analysis(an, a.method, a.smooth, a.min_dur, a.penalty, a.feature)

    out = Path(a.out) / Path(a.video).stem
    out.mkdir(parents=True, exist_ok=True)
    rows = [s.to_dict() for s in segs]
    (out / "segments.json").write_text(json.dumps(
        {"video": a.video, "duration": an.duration, "model": an.model_name,
         "method": a.method, "segments": rows}, indent=2))
    pd.DataFrame([{k: v for k, v in r.items() if k != "alternatives"} for r in rows]
                 ).to_csv(out / "segments.csv", index=False)
    plot_timeline(segs, an.duration, title=f"{Path(a.video).name} ({an.model_name}, {a.method})"
                  ).savefig(out / "timeline.png", dpi=150)

    print(f"{'start':>7} {'end':>7}  {'conf':>5}  label")
    for r in rows:
        print(f"{r['start']:7.1f} {r['end']:7.1f}  {r['confidence']:5.2f}  {r['label']}")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()