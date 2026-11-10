"""
summarize_runs.py — One line per finished run under a directory, best first.

    python src/summarize_runs.py outputs

Runs are ranked by validation average precision, which is what settings should
be chosen on; the test columns are there to be read, not to pick a winner.
"""

from __future__ import annotations

import glob
import json
import os
import sys


def load_runs(root: str) -> list:
    runs = []
    for path in glob.glob(os.path.join(root, "**", "tgn_metrics.json"), recursive=True):
        with open(path) as f:
            metrics = json.load(f)
        if "validation" not in metrics:        # a metrics file from the first version of the model
            continue
        metrics["run"] = os.path.relpath(os.path.dirname(path), root)
        runs.append(metrics)
    return sorted(runs, key=lambda m: -m["validation"]["avg_precision"])


def number(value, pattern: str = "{:.3f}") -> str:
    return "-" if value is None else pattern.format(value)


def main(root: str) -> None:
    runs = load_runs(root)
    if not runs:
        print(f"No finished runs under {root}")
        return

    header = (f"{'run':<28}{'data':<12}{'batch':>6}{'lr':>8}{'weight':>7}{'seed':>5}{'best/ep':>9}"
              f" | {'val AP':>6} | {'test AP':>7}{'F1':>7}{'prec':>7}{'recall':>7}"
              f" | {'AP':>6}{'F1':>7}  (without quiet days) | {'minutes':>7}")
    print(header)
    print("-" * len(header))
    for m in runs:
        config, test = m["config"], m["test"]
        active = m.get("test_active_days", {})
        data = os.path.basename(m["data"]["file"]).replace("_Trans.csv", "")
        print(
            f"{m['run']:<28}{data:<12}{config['batch_size']:>6}{config['lr']:>8.0e}"
            f"{config['pos_weight'] if config['pos_weight'] is not None else 'bal':>7}{config['seed']:>5}"
            f"{str(m['best_epoch']) + '/' + str(len(m['history'])):>9}"
            f" | {number(m['validation']['avg_precision']):>6}"
            f" | {number(test['avg_precision']):>7}{number(test['f1']):>7}"
            f"{number(test['precision']):>7}{number(test['recall']):>7}"
            f" | {number(active.get('avg_precision')):>6}{number(active.get('f1')):>7}{'':>23}"
            f" | {m['train_seconds'] / 60:>7.0f}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs")
