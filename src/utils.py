"""
utils.py — Shared data loading, feature engineering, graph construction, and metrics.

IBM AML HI-Small dataset expected at data/raw/HI-Small_Trans.csv.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data, TemporalData

DEFAULT_TRANS_PATH = os.path.join("data", "raw", "HI-Small_Trans.csv")


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all relevant random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Raw data loading ──────────────────────────────────────────────────────────

def load_transactions(path: str = DEFAULT_TRANS_PATH) -> pd.DataFrame:
    """
    Load and normalise the IBM AML transaction CSV.

    Adds:
      - from_id / to_id : unique account identifiers ("bank_account")
      - timestamp       : parsed datetime, sorted ascending
    """
    df = pd.read_csv(path)
    df.columns = [
        "timestamp", "from_bank", "from_account",
        "to_bank",   "to_account",
        "amount_received", "receiving_currency",
        "amount_paid",     "payment_currency",
        "payment_format",  "is_laundering",
    ]
    # infer_datetime_format is deprecated in pandas ≥ 2.0; format="mixed" is faster
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df["from_id"]   = df["from_bank"].astype(str) + "_" + df["from_account"].astype(str)
    df["to_id"]     = df["to_bank"].astype(str)   + "_" + df["to_account"].astype(str)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_account_labels(df: pd.DataFrame) -> pd.Series:
    """
    Account-level mule label: 1 if the account appears in ANY laundering
    transaction (as sender or receiver), else 0.
    """
    laundering    = df[df["is_laundering"] == 1]
    mule_accounts = set(laundering["from_id"]) | set(laundering["to_id"])
    all_accounts  = set(df["from_id"]) | set(df["to_id"])
    return pd.Series(
        {acc: int(acc in mule_accounts) for acc in all_accounts},
        name="is_mule",
    )


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-account aggregate features used by all four models.

    Scaling is intentionally NOT applied here — LightGBM doesn't need it,
    and GNNs get it via StandardScaler inside build_static_graph.
    """
    df = df.copy()
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    sent = df.groupby("from_id").agg(
        out_tx_count          = ("amount_paid", "count"),
        out_total_amount      = ("amount_paid", "sum"),
        out_mean_amount       = ("amount_paid", "mean"),
        out_std_amount        = ("amount_paid", "std"),
        out_max_amount        = ("amount_paid", "max"),
        out_min_amount        = ("amount_paid", "min"),
        out_unique_recipients = ("to_id", "nunique"),
        out_unique_currencies = ("payment_currency", "nunique"),
        out_unique_formats    = ("payment_format", "nunique"),
        out_unique_hours      = ("hour", "nunique"),
        out_unique_days       = ("day_of_week", "nunique"),
    )

    recv = df.groupby("to_id").agg(
        in_tx_count       = ("amount_received", "count"),
        in_total_amount   = ("amount_received", "sum"),
        in_mean_amount    = ("amount_received", "mean"),
        in_std_amount     = ("amount_received", "std"),
        in_max_amount     = ("amount_received", "max"),
        in_min_amount     = ("amount_received", "min"),
        in_unique_senders = ("from_id", "nunique"),
        in_unique_hours   = ("hour", "nunique"),
        in_unique_days    = ("day_of_week", "nunique"),
    )

    features = sent.join(recv, how="outer").fillna(0)

    features["in_out_amount_ratio"] = (
        features["in_total_amount"] / (features["out_total_amount"] + 1e-9)
    )
    features["in_out_count_ratio"] = (
        features["in_tx_count"] / (features["out_tx_count"] + 1e-9)
    )
    features["fan_in_ratio"] = (
        features["in_unique_senders"] / (features["in_tx_count"] + 1e-9)
    )
    features["fan_out_ratio"] = (
        features["out_unique_recipients"] / (features["out_tx_count"] + 1e-9)
    )
    # Amount spread: high spread relative to mean is a layering signal
    features["out_amount_cv"] = (
        features["out_std_amount"] / (features["out_mean_amount"] + 1e-9)
    )
    features["in_amount_cv"] = (
        features["in_std_amount"] / (features["in_mean_amount"] + 1e-9)
    )

    return features.astype(np.float32)


# ── Static graph construction (GraphSAGE / GAT) ───────────────────────────────

def build_static_graph(df: pd.DataFrame) -> tuple:
    """
    Build a static directed PyG graph with StandardScaler-normalised node features.

    Returns (Data, account_to_idx, scaler).
    """
    all_accounts   = sorted(set(df["from_id"]) | set(df["to_id"]))
    account_to_idx = {acc: i for i, acc in enumerate(all_accounts)}
    n              = len(all_accounts)

    # Node labels — vectorised via reindex instead of per-account loop
    labels = build_account_labels(df)
    y      = torch.tensor(
        labels.reindex(all_accounts, fill_value=0).values, dtype=torch.long
    )

    # Node features — vectorised: reindex aligns by account name, fills missing with 0
    tabular   = engineer_tabular_features(df)
    raw_feats = tabular.reindex(all_accounts, fill_value=0).values.astype(np.float32)

    # StandardScaler: zero mean, unit variance per feature.
    # Without this, amount features (~1e6) dominate gradient flow and GNNs
    # collapse to predicting the majority class.
    scaler     = StandardScaler()
    norm_feats = scaler.fit_transform(raw_feats).astype(np.float32)
    x          = torch.tensor(norm_feats, dtype=torch.float)

    # Edge indices — vectorised via map
    le_curr = LabelEncoder().fit(df["payment_currency"].astype(str))
    le_fmt  = LabelEncoder().fit(df["payment_format"].astype(str))
    df = df.copy()
    df["currency_enc"] = le_curr.transform(df["payment_currency"].astype(str))
    df["format_enc"]   = le_fmt.transform(df["payment_format"].astype(str))

    src_idx    = df["from_id"].map(account_to_idx).values
    dst_idx    = df["to_id"].map(account_to_idx).values
    edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)
    edge_attr  = torch.tensor(
        df[["amount_paid", "currency_enc", "format_enc"]].values, dtype=torch.float
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data, account_to_idx, scaler


# ── Temporal graph construction (TGN) ────────────────────────────────────────

def build_temporal_data(df: pd.DataFrame) -> tuple:
    """
    Build a PyG TemporalData object. Transactions are pre-sorted by timestamp.

    msg features: [log1p(amount_paid), currency_enc, format_enc]
    log1p reduces scale dominance in TGN message aggregation.

    Returns (TemporalData, account_to_idx).
    """
    all_accounts   = sorted(set(df["from_id"]) | set(df["to_id"]))
    account_to_idx = {acc: i for i, acc in enumerate(all_accounts)}

    le_curr = LabelEncoder().fit(df["payment_currency"].astype(str))
    le_fmt  = LabelEncoder().fit(df["payment_format"].astype(str))
    df = df.copy()
    df["currency_enc"] = le_curr.transform(df["payment_currency"].astype(str))
    df["format_enc"]   = le_fmt.transform(df["payment_format"].astype(str))
    df["amount_log"]   = np.log1p(df["amount_paid"])

    # Vectorised index mapping
    src = torch.tensor(df["from_id"].map(account_to_idx).values, dtype=torch.long)
    dst = torch.tensor(df["to_id"].map(account_to_idx).values,   dtype=torch.long)
    t   = torch.tensor(
        (df["timestamp"].astype(np.int64) // 10 ** 9).values, dtype=torch.long
    )
    msg = torch.tensor(
        df[["amount_log", "currency_enc", "format_enc"]].values.astype(np.float32),
        dtype=torch.float,
    )
    y = torch.tensor(df["is_laundering"].values, dtype=torch.long)

    data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
    return data, account_to_idx


# ── Metrics ───────────────────────────────────────────────────────────────────

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Sweep the precision-recall curve and return the threshold that maximises F1.
    Always use this instead of a fixed 0.5 on imbalanced datasets.
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1s  = 2 * (prec * rec) / (prec + rec + 1e-9)
    best = np.argmax(f1s[:-1])   # last entry of prec/rec has no threshold
    return float(thresholds[best])


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold":     round(threshold, 4),
        "auc_roc":       float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "f1":            float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":     float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":        float(recall_score(y_true, y_pred, zero_division=0)),
    }


def print_metrics(metrics: dict, model_name: str = "Model") -> None:
    print(f"\n{'=' * 45}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'=' * 45}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:.4f}")
        else:
            print(f"  {k:<22} {v}")
    print(f"{'=' * 45}\n")


def print_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"  Confusion matrix (threshold={threshold:.4f})")
    print(f"  {'─' * 30}")
    print(f"  TP: {tp:>6,}   FP: {fp:>6,}")
    print(f"  FN: {fn:>6,}   TN: {tn:>6,}")
    print(f"  False positive rate: {fp / (fp + tn + 1e-9):.4f}")
    print(f"  False negative rate: {fn / (fn + tp + 1e-9):.4f}\n")
