from src.metrics import boundary_metrics, edit_score, evaluate, make_match_fn, segment_f1

match = make_match_fn({"A": ["alpha"], "B": ["beta"]})
GT = [{"label": "A", "start": 0, "end": 5}, {"label": "B", "start": 5, "end": 10}]


def test_perfect_prediction():
    pred = [{"label": "alpha thing", "start": 0, "end": 5}, {"label": "beta thing", "start": 5, "end": 10}]
    r = evaluate(pred, GT, 10, match)
    assert r["acc"] == 100 and r["edit"] == 100 and r["f1@50"] == 100
    assert r["boundary_mae_s"] == 0


def test_wrong_labels_and_over_segmentation():
    pred = [{"label": "zzz", "start": 0, "end": 2}, {"label": "alpha", "start": 2, "end": 5},
            {"label": "beta", "start": 5, "end": 10}]
    assert segment_f1(pred, GT, 0.5, match) < 100
    assert edit_score(pred, GT, match) < 100
    assert 0 < evaluate(pred, GT, 10, match)["acc"] < 100


def test_boundary_tolerance():
    pred = [{"label": "alpha", "start": 0, "end": 6.0}, {"label": "beta", "start": 6.0, "end": 10}]
    f1, mae = boundary_metrics(pred, GT, tol=1.5)
    assert f1 == 100 and abs(mae - 1.0) < 1e-9
    f1, _ = boundary_metrics(pred, GT, tol=0.5)
    assert f1 == 0
