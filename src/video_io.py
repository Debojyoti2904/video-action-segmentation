"""Video decoding, resampling and sliding-window extraction."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoData:
    frames: np.ndarray      # (N, size, size, 3) uint8 RGB, sampled at `sample_fps`
    sample_fps: float
    duration: float         # seconds covered by `frames`
    src_fps: float
    truncated: bool         # True if the video was cut at `max_seconds`


def _resize_center_crop(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = size / min(h, w)
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
    rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (nh - size) // 2, (nw - size) // 2
    return rgb[top:top + size, left:left + size]


def load_video(path: str, sample_fps: float = 8.0, size: int = 224,
               max_seconds: float = 600.0) -> VideoData:
    """Decode sequentially, resample to `sample_fps`, resize short side + center crop."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or np.isnan(src_fps) or src_fps < 1:
        src_fps = 30.0

    frames, idx, next_t, step, truncated = [], 0, 0.0, 1.0 / sample_fps, False
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = idx / src_fps
        if t > max_seconds:
            truncated = True
            break
        while t + 1e-9 >= next_t:          # handles src_fps < sample_fps (duplicates)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(_resize_center_crop(rgb, size))
            next_t += step
        idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from: {path}")
    arr = np.stack(frames)
    return VideoData(arr, sample_fps, len(arr) / sample_fps, src_fps, truncated)


def make_windows(n_frames: int, clip_len: int = 16, hop: int = 4) -> list[int]:
    """Start indices of sliding windows; the tail is always covered."""
    if n_frames <= clip_len:
        return [0]
    starts = list(range(0, n_frames - clip_len + 1, hop))
    if starts[-1] + clip_len < n_frames:
        starts.append(n_frames - clip_len)
    return starts


def get_clip(frames: np.ndarray, start: int, clip_len: int = 16) -> np.ndarray:
    clip = frames[start:start + clip_len]
    if len(clip) < clip_len:               # pad very short videos by repeating last frame
        pad = np.repeat(clip[-1:], clip_len - len(clip), axis=0)
        clip = np.concatenate([clip, pad], axis=0)
    return clip
