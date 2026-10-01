"""Segmentation metrics. Segments are dicts: {"start", "end", "label"} (seconds)."""
from __future__ import annotations

import numpy as np


def _dicts(segs):
    return [s if isinstance(s, dict) else s.to_dict() for s in segs]


def make_match_fn(keyword_map: dict | None = None):
    """Label matcher. GT label -> keywords; a prediction matches if it contains any."""
    keyword_map = keyword_map or {}

    def match(pred_label: str, gt_label: str) -> bool:
        kws = keyword_map.get(gt_label) or [gt_label.lower()]
        p = pred_label.lower()
        return any(k.lower() in p for k in kws)

    return match


def _iou(a, b):
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def _label_at(segs, t):
    for s in segs:
        if s["start"] <= t < s["end"]:
            return s["label"]
    return None


def frame_accuracy(pred, gt, duration, match, dt=0.1):
    pred, gt = _dicts(pred), _dicts(gt)
    ts = np.arange(0, duration, dt) + dt / 2
    ok = tot = 0
    for t in ts:
        g = _label_at(gt, t)
        if g is None:
            continue
        tot += 1
        p = _label_at(pred, t)
        ok += int(p is not None and match(p, g))
    return 100.0 * ok / max(tot, 1)


def segment_f1(pred, gt, thr, match):
    pred, gt = _dicts(pred), _dicts(gt)
    used, tp = set(), 0
    for p in pred:
        best, bj = 0.0, -1
        for j, g in enumerate(gt):
            if j in used or not match(p["label"], g["label"]):
                continue
            i = _iou(p, g)
            if i > best:
                best, bj = i, j
        if bj >= 0 and best >= thr:
            tp += 1
            used.add(bj)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    return 100.0 * (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)


def edit_score(pred, gt, match):
    pred, gt = _dicts(pred), _dicts(gt)
    m, n = len(pred), len(gt)
    D = np.zeros((m + 1, n + 1))
    D[:, 0], D[0, :] = np.arange(m + 1), np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if match(pred[i - 1]["label"], gt[j - 1]["label"]) else 1
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost)
    return 100.0 * (1 - D[m, n] / max(m, n, 1))


def boundary_metrics(pred, gt, tol=1.5):
    """Label-free boundary quality: F1 within +-tol seconds and mean abs error."""
    pb = [s["start"] for s in _dicts(pred)[1:]]
    gb = [s["start"] for s in _dicts(gt)[1:]]
    if not pb and not gb:
        return 100.0, 0.0
    used, errs = set(), []
    for g in gb:
        cands = [(abs(p - g), k) for k, p in enumerate(pb) if k not in used and abs(p - g) <= tol]
        if cands:
            e, k = min(cands)
            used.add(k)
            errs.append(e)
    tp = len(errs)
    prec = tp / len(pb) if pb else 0.0
    rec = tp / len(gb) if gb else 0.0
    f1 = 100.0 * (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return f1, (float(np.mean(errs)) if errs else float("nan"))


def evaluate(pred, gt, duration, match, tol=1.5):
    bf1, bmae = boundary_metrics(pred, gt, tol)
    return {
        "acc": frame_accuracy(pred, gt, duration, match),
        "edit": edit_score(pred, gt, match),
        "f1@10": segment_f1(pred, gt, 0.10, match),
        "f1@25": segment_f1(pred, gt, 0.25, match),
        "f1@50": segment_f1(pred, gt, 0.50, match),
        f"boundary_f1@{tol}s": bf1,
        "boundary_mae_s": bmae,
        "n_pred": len(pred),
        "n_gt": len(gt),
    }
