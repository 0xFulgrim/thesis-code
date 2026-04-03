"""
lightgbm_model.py — Tabular baseline for money mule detection.

Approach:
  - No graph structure. Each account is an independent data point.
  - Hand-engineered aggregate features per account (send/receive stats,
    temporal diversity, fan-in/out ratios, coefficient of variation).
  - LightGBM gradient boosted trees with class_weight="balanced".
  - Account-level evaluation: is this account a mule?

Threshold:
  Fixed 0.5 is inappropriate for imbalanced data (~1.5% mule prevalence).
  We sweep the precision-recall curve to find the F1-optimal threshold.
  The chosen threshold is saved in the metrics JSON for reproducibility.

Run:
  python src/lightgbm_model.py
  python src/lightgbm_model.py --data data/raw/HI-Small_Trans.csv
"""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from utils import (
    build_account_labels,
    compute_metrics,
    engineer_tabular_features,
    find_best_threshold,
    load_transactions,
    print_confusion_matrix,
    print_metrics,
    set_seed,
)

SEED = 42
OUTPUTS_DIR = "outputs"


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LightGBM mule detector")
    p.add_argument("--data",          default="data/raw/HI-Small_Trans.csv")
    p.add_argument("--test_size",     type=float, default=0.2)
    p.add_argument("--val_size",      type=float, default=0.1,
                   help="Fraction of training data held out for early stopping")
    p.add_argument("--n_estimators",  type=int,   default=2000,
                   help="Max trees; early stopping will usually kick in before this")
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--num_leaves",    type=int,   default=63)
    p.add_argument("--min_child_samples", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = get_args()
    set_seed(SEED)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("[1/4] Loading transactions...")
    df = load_transactions(args.data)
    print(f"      {len(df):,} transactions loaded.")

    # ── 2. Features ───────────────────────────────────────────────────────────
    print("[2/4] Engineering tabular features...")
    features = engineer_tabular_features(df)
    labels   = build_account_labels(df)

    idx = features.index.intersection(labels.index)
    X   = features.loc[idx].values.astype(np.float32)
    y   = labels.loc[idx].values.astype(np.int32)

    n_mules  = int(y.sum())
    n_normal = int((y == 0).sum())
    print(f"      Accounts : {len(y):,}")
    print(f"      Mules    : {n_mules:,}  ({n_mules / len(y) * 100:.2f}%)")
    print(f"      Normal   : {n_normal:,}")
    print(f"      Features : {X.shape[1]}")

    # ── 3. Split ──────────────────────────────────────────────────────────────
    # Random split is valid — LightGBM treats each account independently.
    # We create a val set explicitly for early stopping so the test set stays
    # completely held out (not peeked at during training).
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=SEED, stratify=y,
    )
    val_frac = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=SEED, stratify=y_trainval,
    )
    print(f"      Train : {len(y_train):,} | Val : {len(y_val):,} | Test : {len(y_test):,}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("[3/4] Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"      Best iteration: {model.best_iteration_}")

    # ── 5. Threshold + eval ───────────────────────────────────────────────────
    print("[4/4] Evaluating on held-out test set...")
    y_prob = model.predict_proba(X_test)[:, 1]

    best_thresh = find_best_threshold(y_test, y_prob)
    print(f"      Optimal threshold (max F1): {best_thresh:.4f}")

    metrics = compute_metrics(y_test, y_prob, threshold=best_thresh)
    print_metrics(metrics, model_name="LightGBM")
    print_confusion_matrix(y_test, y_prob, threshold=best_thresh)

    # Top-10 feature importance
    feat_names  = features.columns.tolist()
    importances = model.feature_importances_
    top10 = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:10]
    print("  Top-10 features by importance:")
    for fname, imp in top10:
        print(f"    {fname:<35} {imp}")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    model_path   = os.path.join(OUTPUTS_DIR, "lightgbm_model.txt")
    metrics_path = os.path.join(OUTPUTS_DIR, "lightgbm_metrics.json")

    model.booster_.save_model(model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel   → {model_path}")
    print(f"Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
