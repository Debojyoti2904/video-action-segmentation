"""End-to-end test on a synthetic video with a stub model (no weights needed)."""
import cv2
import numpy as np

from src.pipeline import analyze, segment_analysis


class StubModel:
    name = "stub"
    labels = ["red", "green", "blue"]

    def predict(self, clips):
        probs, emb = [], []
        for c in clips:
            m = c.reshape(-1, 3).mean(0) / 255.0           # mean RGB
            p = np.exp(6 * m); probs.append(p / p.sum()); emb.append(m)
        return np.array(probs), np.array(emb)


def make_video(path, colors=((255, 0, 0), (0, 255, 0), (0, 0, 255)), sec=6, fps=15):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    rng = np.random.default_rng(0)
    for rgb in colors:
        for _ in range(sec * fps):
            f = np.zeros((120, 160, 3), np.uint8); f[:] = rgb[::-1]
            f = np.clip(f + rng.integers(-10, 10, f.shape), 0, 255).astype(np.uint8)
            vw.write(f)
    vw.release()


def test_three_color_video(tmp_path):
    path = tmp_path / "v.avi"
    make_video(path)
    a = analyze(str(path), StubModel())
    assert abs(a.duration - 18.0) < 0.3
    for method in ("merge", "changepoint"):
        segs = segment_analysis(a, method)
        assert [s.label for s in segs] == ["red", "green", "blue"]
        assert abs(segs[1].start - 6) < 1.5 and abs(segs[2].start - 12) < 1.5
        assert all(0 < s.confidence <= 1 for s in segs)
