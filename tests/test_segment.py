import numpy as np

from src.segment import segment


def synthetic(n_per=16, noise=0.05, seed=0):
    """3 'actions' of n_per windows each, hop 0.5s, window 2s. Returns probs, embeds, times."""
    rng = np.random.default_rng(seed)
    C, D = 6, 32
    protos = rng.normal(size=(3, D))
    P, E = [], []
    for k, cls in enumerate([0, 3, 5]):
        for _ in range(n_per):
            p = np.full(C, 0.02)
            p[cls] = 0.9
            P.append(p / p.sum() + rng.normal(0, noise * 0.1, C).clip(-0.01))
            E.append(protos[k] + rng.normal(0, noise * 3, D))
    P = np.abs(np.array(P)); P /= P.sum(1, keepdims=True)
    n = len(P)
    ws = np.arange(n) * 0.5
    return P, np.array(E), ws, ws + 2.0, n * 0.5 + 1.5, [f"c{i}" for i in range(C)]


def _check(segs, truth=(8.0 + 0.75, 16.0 + 1.5)):
    assert [s.label for s in segs] == ["c0", "c3", "c5"]
    bounds = [s.start for s in segs[1:]]
    for b, t in zip(bounds, (8.75, 17.5)):
        assert abs(b - t) <= 1.5


def test_merge_finds_three_segments():
    P, E, ws, we, dur, names = synthetic()
    _check(segment(P, E, ws, we, dur, names, method="merge", step=0.5))


def test_changepoint_embed_and_probs():
    P, E, ws, we, dur, names = synthetic()
    for feat in ("embed", "probs"):
        _check(segment(P, E, ws, we, dur, names, method="changepoint", step=0.5, feature=feat))


def test_single_action_gives_single_segment():
    P, E, ws, we, dur, names = synthetic()
    P[:] = 0.02; P[:, 2] = 0.9; P /= P.sum(1, keepdims=True)
    E = np.random.default_rng(1).normal(size=E.shape) * 0.01 + E[0]
    for m in ("merge", "changepoint"):
        segs = segment(P, E, ws, we, dur, names, method=m, step=0.5)
        assert len(segs) == 1 and segs[0].label == "c2"
