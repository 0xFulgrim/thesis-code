import numpy as np
import pytest

from evaluation import best_f1_threshold, classification_metrics, detection_latency


def test_threshold_maximises_f1():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.35, 0.8])
    # 0.35 flags three, two of them right: F1 = 0.8, better than any other cut.
    assert best_f1_threshold(y, score) == pytest.approx(0.35)


def test_threshold_needs_positives():
    with pytest.raises(ValueError):
        best_f1_threshold(np.zeros(4, dtype=int), np.linspace(0, 1, 4))


def test_metrics_at_threshold():
    y = np.array([0, 0, 1, 1, 1, 0])
    score = np.array([0.1, 0.6, 0.7, 0.2, 0.9, 0.3])
    m = classification_metrics(y, score, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert 0.0 <= m["auc_roc"] <= 1.0 and 0.0 <= m["avg_precision"] <= 1.0


def test_metrics_with_one_class_and_no_alerts():
    m = classification_metrics(np.zeros(3, dtype=int), np.array([0.1, 0.2, 0.3]), threshold=0.5)
    assert m["auc_roc"] is None and m["avg_precision"] is None
    assert (m["precision"], m["recall"], m["f1"]) == (0.0, 0.0, 0.0)


def test_detection_latency():
    A, B, C, D, E, F, G, H = range(8)
    #        src dst   t  laundering  flagged
    rows = [
        (A, B, 10, 0, 1),   # alert on A before it launders: not a detection
        (A, C, 20, 1, 0),   # onset of A and C
        (C, D, 30, 0, 1),   # C detected, latency 10; D is not a mule
        (A, A, 50, 0, 1),   # A detected, latency 30 (a self-transfer)
        (E, F, 60, 1, 1),   # E and F: onset and detection together, latency 0
        (G, H, 70, 1, 0),   # G and H are never flagged
    ]
    src, dst, t, y, flagged = (np.array(column) for column in zip(*rows, strict=True))
    result = detection_latency(src, dst, t, y, flagged.astype(bool))

    assert result["mule_accounts"] == 6
    assert result["detected_accounts"] == 4
    assert result["detection_rate"] == pytest.approx(4 / 6)
    assert result["mean_latency_seconds"] == pytest.approx(10.0)
    assert result["median_latency_seconds"] == pytest.approx(5.0)
    assert result["zero_latency_accounts"] == 2
    assert result["flagged_before_onset"] == 1


def test_detection_latency_without_mules_or_detections():
    src, dst, t = np.array([0, 1]), np.array([1, 2]), np.array([5, 6])
    none = detection_latency(src, dst, t, np.array([0, 0]), np.array([True, False]))
    assert none["mule_accounts"] == 0 and none["detection_rate"] is None

    missed = detection_latency(src, dst, t, np.array([0, 1]), np.array([False, False]))
    assert missed["mule_accounts"] == 2 and missed["detected_accounts"] == 0
    assert missed["median_latency_seconds"] is None
