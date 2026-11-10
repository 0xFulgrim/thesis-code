"""
ensemble.py — Average the scores of several runs and evaluate the average.

    python src/ensemble.py outputs/seed1 outputs/seed2 outputs/seed3

The runs must come from the same data and split and have been made with
--save_scores. As for a single run, the threshold is chosen on the validation
scores (here their average) and then applied to the test scores.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from evaluation import (
    best_f1_threshold,
    classification_metrics,
    detection_latency,
    format_report,
)


def average_scores(runs: list, name: str) -> dict:
    """The arrays of the first run, with `score` replaced by the mean over all runs."""
    loaded = [np.load(os.path.join(run, name)) for run in runs]
    first = {key: loaded[0][key] for key in loaded[0].files}
    for other in loaded[1:]:
        if not np.array_equal(other["label"], first["label"]):
            raise SystemExit(f"{name}: the runs were not scored on the same transactions")
    first["score"] = np.mean([scores["score"] for scores in loaded], axis=0)
    return first


def evaluate(runs: list) -> dict:
    val = average_scores(runs, "tgn_val_scores.npz")
    test = average_scores(runs, "tgn_test_scores.npz")

    threshold = best_f1_threshold(val["label"], val["score"])
    flagged = test["score"] >= threshold
    active = test["active_day"]

    def latency(mask):
        return detection_latency(test["src"][mask], test["dst"][mask], test["t"][mask],
                                 test["label"][mask], flagged[mask])

    everything = np.ones_like(active)
    return {
        "runs": runs,
        "validation": classification_metrics(val["label"], val["score"], threshold),
        "test": classification_metrics(test["label"], test["score"], threshold),
        "detection_latency": latency(everything),
        "test_active_days": classification_metrics(test["label"][active], test["score"][active], threshold),
        "detection_latency_active_days": latency(active),
    }


def main(runs: list) -> dict:
    if len(runs) < 2:
        raise SystemExit("Give at least two run directories.")
    results = evaluate(runs)
    print(format_report(f"Ensemble of {len(runs)} runs — validation", results["validation"]))
    print(format_report("Ensemble — test", results["test"], results["detection_latency"]))
    print(format_report("Ensemble — test without quiet days",
                        results["test_active_days"], results["detection_latency_active_days"]))
    return results


if __name__ == "__main__":
    output = main(sys.argv[1:])
    with open("ensemble_metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Metrics → ensemble_metrics.json")
