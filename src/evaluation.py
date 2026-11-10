"""
evaluation.py — Threshold selection, classification metrics, detection latency.

Everything here works on NumPy arrays, so it can be reused on saved scores.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

SECONDS_PER_DAY = 86_400


# ── Quiet days ────────────────────────────────────────────────────────────────

def quiet_day_mask(t: np.ndarray, fraction: float = 0.01) -> np.ndarray:
    """
    Marks the transactions that fall on a quiet calendar day: one with fewer
    than `fraction` of the transactions of a typical day.

    The IBM AML files end with a long run of such days. Normal activity has
    stopped by then and most of what is left is laundering (in HI-Small, 655
    of the last 1,108 transactions), so those days flatter any model. They all
    land in the test split; reporting the test metrics with and without them
    shows how much of a result they account for.

    The typical day is the one the median transaction falls on. That stays a
    busy day however many quiet days there are, because they hold so little.
    """
    day = t // SECONDS_PER_DAY
    day = day - day.min()
    per_day = np.bincount(day)
    typical = np.median(per_day[day])
    return (per_day < fraction * typical)[day]


# ── Threshold ─────────────────────────────────────────────────────────────────

def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    The score threshold that maximises F1 on the given labels.

    Call this on the validation split and apply the result to the test split;
    choosing it on the test labels makes F1, precision and recall optimistic.
    """
    if not np.any(y_true):
        raise ValueError("no positive labels: cannot select a threshold")
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    # The last precision/recall pair has no threshold attached to it.
    return float(thresholds[np.argmax(f1[:-1])])


# ── Classification metrics ────────────────────────────────────────────────────

def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """
    Ranking metrics (ROC AUC, average precision) and the metrics at `threshold`.

    The ranking metrics are None when the labels contain a single class.
    """
    positive = np.asarray(y_true).astype(bool)
    flagged = np.asarray(y_score) >= threshold

    tp = int(np.sum(flagged & positive))
    fp = int(np.sum(flagged & ~positive))
    fn = int(np.sum(~flagged & positive))
    tn = int(positive.size - tp - fp - fn)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    both_classes = positive.any() and not positive.all()
    return {
        "auc_roc": float(roc_auc_score(positive, y_score)) if both_classes else None,
        "avg_precision": float(average_precision_score(positive, y_score)) if both_classes else None,
        "threshold": float(threshold),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ── Detection latency ─────────────────────────────────────────────────────────

def detection_latency(
    src: np.ndarray,
    dst: np.ndarray,
    t: np.ndarray,
    y_true: np.ndarray,
    flagged: np.ndarray,
) -> dict:
    """
    How quickly mule accounts are caught once they start laundering.

    Works on one evaluation stream (parallel arrays, one entry per transaction).
    A flagged transaction raises an alert on both of its accounts.

      onset     : time of an account's first laundering transaction in the stream.
      detection : the account's first alert at or after its onset.
      latency   : detection time minus onset, in seconds; never negative.

    An alert raised before the onset does not count as a detection, because
    nothing had been laundered yet. `flagged_before_onset` counts the mule
    accounts that had such an early alert, whether or not they were also
    detected later.
    """
    account, index = np.unique(np.concatenate([src, dst]), return_inverse=True)
    time = np.concatenate([t, t]).astype(np.int64)
    laundering = np.concatenate([y_true, y_true]).astype(bool)
    alert = np.concatenate([flagged, flagged]).astype(bool)

    never = np.iinfo(np.int64).max
    onset = np.full(account.size, never, dtype=np.int64)
    np.minimum.at(onset, index[laundering], time[laundering])
    is_mule = onset < never

    onset_of_row = onset[index]
    counts = alert & (time >= onset_of_row)       # onset is `never` for non-mules
    detection = np.full(account.size, never, dtype=np.int64)
    np.minimum.at(detection, index[counts], time[counts])
    detected = detection < never

    early = alert & (time < onset_of_row) & is_mule[index]
    latency = (detection - onset)[detected]

    mules = int(is_mule.sum())
    caught = int(detected.sum())
    return {
        "mule_accounts": mules,
        "detected_accounts": caught,
        "detection_rate": caught / mules if mules else None,
        "mean_latency_seconds": float(latency.mean()) if caught else None,
        "median_latency_seconds": float(np.median(latency)) if caught else None,
        "p90_latency_seconds": float(np.percentile(latency, 90)) if caught else None,
        "zero_latency_accounts": int(np.sum(latency == 0)),
        "flagged_before_onset": int(np.unique(index[early]).size),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def format_report(title: str, metrics: dict, latency: dict | None = None) -> str:
    """A fixed-width text block for the console."""
    def number(value, pattern="{:.4f}"):
        return "n/a" if value is None else pattern.format(value)

    lines = [
        "=" * 52,
        f"  {title}",
        "=" * 52,
        f"  ROC AUC            {number(metrics['auc_roc'])}",
        f"  Average precision  {number(metrics['avg_precision'])}",
        f"  Threshold          {metrics['threshold']:.6f}",
        f"  F1                 {metrics['f1']:.4f}",
        f"  Precision          {metrics['precision']:.4f}",
        f"  Recall             {metrics['recall']:.4f}",
        f"  TP {metrics['tp']:,}   FP {metrics['fp']:,}   FN {metrics['fn']:,}   TN {metrics['tn']:,}",
    ]
    if latency is not None:
        lines += [
            "-" * 52,
            f"  Mule accounts      {latency['mule_accounts']:,}",
            (f"  Detected           {latency['detected_accounts']:,}"
             f"  ({number(latency['detection_rate'], '{:.1%}')})"),
            f"  Median latency     {number(latency['median_latency_seconds'], '{:,.0f} s')}",
            f"  Mean latency       {number(latency['mean_latency_seconds'], '{:,.0f} s')}",
            f"  90th percentile    {number(latency['p90_latency_seconds'], '{:,.0f} s')}",
            f"  Alerted before onset {latency['flagged_before_onset']:,}",
        ]
    lines.append("=" * 52)
    return "\n".join(lines)
