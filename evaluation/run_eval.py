"""Evaluate segmentation methods on stitched videos (X-CLIP inference only).

    python -m evaluation.run_eval --manifest eval_data/manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import evaluate, make_match_fn
from src.model import load_model
from src.pipeline import analyze, segment_analysis


def run(manifest, model, match, configs, tol=1.5):
    rows = []
    for item in manifest:
        a = analyze(item["video"], model)            # expensive step, done once per video
        for name, kw in configs.items():
            pred = segment_analysis(a, **kw)
            r = evaluate(pred, item["segments"], item["duration"], match, tol)
            rows.append({"video": Path(item["video"]).name, "config": name, **r})
        print(f"done {item['video']}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="eval_data/manifest.json")
    ap.add_argument("--keywords", default="evaluation/label_keywords.json")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--tol", type=float, default=1.5)
    a = ap.parse_args()

    manifest = json.loads(Path(a.manifest).read_text())
    match = make_match_fn(json.loads(Path(a.keywords).read_text()))
    
    # Dynamically extract all unique labels from the dataset to feed into X-CLIP
    labels = sorted({s["label"] for m in manifest for s in m["segments"]})
    model = load_model(labels=labels)

    configs = {
        "merge (smooth=2s)":          dict(method="merge", smooth_sec=2.0),
        "merge (no smoothing)":       dict(method="merge", smooth_sec=0.0),
        "changepoint (embed)":        dict(method="changepoint", feature="embed"),
        "changepoint (probs)":        dict(method="changepoint", feature="probs"),
    }
    df = run(manifest, model, match, configs, a.tol)
    summary = df.drop(columns=["video"]).groupby("config").mean(numeric_only=True).round(2)
    
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "per_video.csv", index=False)
    summary.to_csv(Path(a.out) / "summary.csv")
    print("\n" + summary.to_string())


if __name__ == "__main__":
    main()