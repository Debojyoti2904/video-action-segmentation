"""End-to-end: video file -> windows -> model -> cached Analysis -> segments."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .segment import segment as _segment
from .video_io import get_clip, load_video, make_windows


@dataclass
class Analysis:
    probs: np.ndarray            # (W, C)
    embeds: np.ndarray           # (W, D)
    win_start: np.ndarray        # seconds
    win_end: np.ndarray          # seconds
    duration: float
    hop_sec: float
    labels: list
    thumbs: np.ndarray           # (T, 160, 160, 3), one per second
    model_name: str
    truncated: bool = False


def analyze(video_path, model, sample_fps=8.0, clip_len=16, hop_frames=4,
            batch_size=8, max_seconds=600.0, progress_cb=None) -> Analysis:
    """The only expensive step. Cache the result and re-segment as often as you like."""
    vd = load_video(video_path, sample_fps=sample_fps, max_seconds=max_seconds)
    starts = make_windows(len(vd.frames), clip_len, hop_frames)
    probs, embeds = [], []
    for i in range(0, len(starts), batch_size):
        batch = starts[i:i + batch_size]
        p, e = model.predict([get_clip(vd.frames, s, clip_len) for s in batch])
        probs.append(p)
        embeds.append(e)
        if progress_cb:
            progress_cb(min(1.0, (i + len(batch)) / len(starts)))

    st = np.array(starts, dtype=float)
    win_start = st / sample_fps
    win_end = np.minimum((st + clip_len) / sample_fps, vd.duration)
    step = max(1, int(round(sample_fps)))
    thumbs = np.stack([cv2.resize(f, (160, 160)) for f in vd.frames[::step]])
    return Analysis(np.concatenate(probs), np.concatenate(embeds), win_start, win_end,
                    vd.duration, hop_frames / sample_fps, list(model.labels), thumbs,
                    getattr(model, "name", "model"), vd.truncated)


def segment_analysis(a: Analysis, method="changepoint", smooth_sec=2.0, min_dur=2.0,
                     penalty_scale=4.0, feature="embed"):
    return _segment(a.probs, a.embeds, a.win_start, a.win_end, a.duration, a.labels,
                    method=method, step=a.hop_sec, smooth_sec=smooth_sec,
                    min_dur=min_dur, penalty_scale=penalty_scale, feature=feature)
