"""End-to-end runs on the synthetic file. A few seconds each on a CPU."""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from tgn_model import main

SMALL = [
    "--no_cache",
    "--memory_dim", "16", "--time_dim", "8", "--embedding_dim", "16",
    "--neighbor_size", "5", "--batch_size", "50", "--lr", "1e-3",
]


def run(csv_path, out_dir, *extra, device="cpu"):
    return main(["--data", csv_path, "--out_dir", str(out_dir), "--device", device, *SMALL, *extra])


def reject_constant(name):
    raise AssertionError(f"metrics file contains {name}, which is not valid JSON")


def test_end_to_end(csv_path, tmp_path):
    results = run(csv_path, tmp_path, "--epochs", "4", "--save_scores")

    for name in ("tgn_metrics.json", "tgn_model.pt", "tgn_checkpoint.pt", "tgn_test_scores.npz"):
        assert os.path.exists(tmp_path / name), name

    with open(tmp_path / "tgn_metrics.json") as f:
        saved = json.load(f, parse_constant=reject_constant)
    assert saved["test"] == results["test"]
    assert len(saved["history"]) == 4
    assert 1 <= saved["best_epoch"] <= 4

    # The threshold comes from validation and is the one applied to the test split.
    assert saved["test"]["threshold"] == saved["validation"]["threshold"]
    for split in ("validation", "test"):
        assert 0.0 <= saved[split]["avg_precision"] <= 1.0
    counts = saved["test"]
    assert counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"] == \
        saved["data"]["splits"]["test"]["transactions"]

    # The synthetic file has no quiet days, so leaving them out changes nothing.
    assert saved["data"]["quiet_days_in_test"] == {"transactions": 0, "laundering": 0}
    assert saved["test_active_days"] == saved["test"]
    assert saved["detection_latency_active_days"] == saved["detection_latency"]

    scores = np.load(tmp_path / "tgn_test_scores.npz")
    assert scores["score"].shape == scores["label"].shape == scores["t"].shape
    assert int((scores["score"] >= scores["threshold"]).sum()) == counts["tp"] + counts["fp"]

    # The planted signal is learnable: after four epochs the ranking should be
    # well above the laundering rate, which is what a random scorer gets.
    test_split = saved["data"]["splits"]["test"]
    base_rate = test_split["laundering"] / test_split["transactions"]
    assert saved["test"]["avg_precision"] > 3 * base_rate


def test_resume_matches_an_uninterrupted_run(csv_path, tmp_path):
    straight = run(csv_path, tmp_path / "straight", "--epochs", "3")

    run(csv_path, tmp_path / "resumed", "--epochs", "2")
    resumed = run(csv_path, tmp_path / "resumed", "--epochs", "3", "--resume")

    assert len(resumed["history"]) == 3
    for a, b in zip(straight["history"], resumed["history"], strict=True):
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-6)
        assert a["val_avg_precision"] == pytest.approx(b["val_avg_precision"], rel=1e-6)
    assert straight["best_epoch"] == resumed["best_epoch"]
    assert straight["test"]["avg_precision"] == pytest.approx(resumed["test"]["avg_precision"], rel=1e-6)
    assert straight["test"]["tp"] == resumed["test"]["tp"]


def test_resume_of_a_finished_run_only_re_evaluates(csv_path, tmp_path):
    first = run(csv_path, tmp_path, "--epochs", "2")
    again = run(csv_path, tmp_path, "--epochs", "2", "--resume")
    assert len(again["history"]) == 2
    assert again["test"]["avg_precision"] == pytest.approx(first["test"]["avg_precision"], rel=1e-6)
    assert again["test"]["tp"] == first["test"]["tp"]
    assert again["detection_latency"] == first["detection_latency"]


def test_resume_rejects_changed_settings(csv_path, tmp_path):
    run(csv_path, tmp_path, "--epochs", "1")
    with pytest.raises(SystemExit, match="memory_dim"):
        run(csv_path, tmp_path, "--memory_dim", "32", "--epochs", "2", "--resume")


def test_ablation_and_truncation_run(csv_path, tmp_path):
    results = run(csv_path, tmp_path, "--epochs", "1", "--no_txn_features",
                  "--max_events", "2000", "--pos_weight", "5", "--max_grad_norm", "1.0")
    assert results["data"]["transactions"] == 2000
    assert results["config"]["pos_weight"] == 5
    assert results["config"]["max_grad_norm"] == 1.0


def test_a_diverging_run_stops_with_a_clear_error(csv_path, tmp_path):
    with pytest.raises(FloatingPointError, match="no longer finite"):
        run(csv_path, tmp_path, "--epochs", "1", "--lr", "1e30")


def test_quiet_days_are_reported_separately(transactions, tmp_path):
    # Append a run of nearly empty days in which almost everything is laundering,
    # as at the end of the real files.
    from conftest import write_csv
    tail = transactions.iloc[:40].copy()
    start = np.datetime64("2022-09-15T10:00")
    tail["timestamp"] = [
        str(start + np.timedelta64(i * 13, "h")).replace("-", "/").replace("T", " ") for i in range(40)
    ]
    tail["is_laundering"] = [1] * 30 + [0] * 10
    path = tmp_path / "Tail_Trans.csv"
    write_csv(pd.concat([transactions, tail]), path)

    results = main(["--data", str(path), "--out_dir", str(tmp_path / "out"), "--device", "cpu",
                    *SMALL, "--epochs", "1"])
    assert results["data"]["quiet_days_in_test"] == {"transactions": 40, "laundering": 30}
    n_test = results["data"]["splits"]["test"]["transactions"]
    counts = results["test_active_days"]
    assert counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"] == n_test - 40
    assert results["detection_latency_active_days"]["mule_accounts"] <= \
        results["detection_latency"]["mule_accounts"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_end_to_end_on_gpu(csv_path, tmp_path):
    """The same pipeline on the GPU, including an interrupted-and-resumed run."""
    run(csv_path, tmp_path, "--epochs", "1", device="cuda")
    results = run(csv_path, tmp_path, "--epochs", "2", "--resume", "--save_scores", device="cuda")
    assert len(results["history"]) == 2
    assert results["config"]["device_used"].startswith("cuda")
    assert 0.0 <= results["test"]["avg_precision"] <= 1.0
    assert results["environment"]["gpu"] is not None
