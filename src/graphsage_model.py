"""
graphsage_model.py — Static inductive GNN baseline for money mule detection.

Approach:
  - Accounts are nodes; each transaction is a directed edge.
  - Node features are the same aggregate statistics as the LightGBM baseline,
    normalised with StandardScaler (applied inside build_static_graph).
    This isolates the effect of adding graph structure vs. tabular features alone.
  - GraphSAGE (Hamilton et al. 2017): inductive, 2-hop sampled aggregation.
    Two SAGEConv layers + BatchNorm + linear classification head.
  - Mini-batch training via NeighborLoader.
  - Validation split for early stopping — test set never touched during training.

Improvements over v2:
  - BatchNorm after each SAGEConv layer: stabilises training on heterogeneous
    account graphs where node degree variance is very high.
  - Gradient clipping (max norm 1.0): prevents exploding gradients, which can
    occur silently when a few high-degree hub nodes dominate the gradient signal.
  - ReduceLROnPlateau scheduler: decays LR when val AUC stops improving.
  - Explicit val split: early stopping based on val AUC-ROC, not train loss.
    This is important because train loss can keep decreasing while the model
    overfits on the majority class.

Run:
  python src/graphsage_model.py
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

from utils import (
    build_static_graph,
    compute_metrics,
    find_best_threshold,
    load_transactions,
    print_confusion_matrix,
    print_metrics,
    set_seed,
)

SEED = 42
OUTPUTS_DIR = "outputs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(torch.nn.Module):
    """
    Focal Loss (Lin et al. 2017) for extreme class imbalance.

    Down-weights easy negatives by (1 - p_t)^gamma, forcing the model to
    focus on hard positives (mule accounts that look like normal ones).

    alpha : weight for the positive class (0.25 is the paper default).
    gamma : focusing strength. gamma=0 → standard BCE. gamma=2 → paper default.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)[:, 1]
        # Clamp: softmax can output exact 0/1 → log(0) = -inf → NaN loss
        probs = probs.clamp(1e-7, 1 - 1e-7)
        bce   = F.binary_cross_entropy(probs, targets.float(), reduction="none")
        pt    = torch.where(targets == 1, probs, 1 - probs)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ── Model ─────────────────────────────────────────────────────────────────────

class GraphSAGE(torch.nn.Module):
    """
    Two-layer GraphSAGE with BatchNorm and a linear classification head.

    Architecture:
      SAGEConv(in → hidden) → BatchNorm → ReLU → Dropout
      SAGEConv(hidden → hidden) → BatchNorm → ReLU → Dropout
      Linear(hidden → 2)

    BatchNorm is placed after the convolution and before the activation,
    following standard practice for GNNs with high-variance node degrees.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels,     hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn1   = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2   = torch.nn.BatchNorm1d(hidden_channels)
        self.lin   = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x, edge_index)).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(self.conv2(x, edge_index)).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


# ── Training / eval loops ─────────────────────────────────────────────────────

def train_epoch(
    model: GraphSAGE,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    criterion: FocalLoss,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss  = 0.0
    n_processed = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index)[: batch.batch_size]
        y    = batch.y[: batch.batch_size]
        loss = criterion(out, y)
        loss.backward()
        # Clip gradients: high-degree hub nodes produce large gradient signals
        # that can destabilise training on graphs with skewed degree distributions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss  += float(loss) * batch.batch_size
        n_processed += batch.batch_size
    return total_loss / max(n_processed, 1)


@torch.no_grad()
def evaluate(model: GraphSAGE, loader: NeighborLoader) -> tuple:
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch  = batch.to(device)
        out    = model(batch.x, batch.edge_index)[: batch.batch_size]
        y      = batch.y[: batch.batch_size]
        probs  = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ── CLI args ──────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraphSAGE mule detector")
    p.add_argument("--data",            default="data/raw/HI-Small_Trans.csv")
    p.add_argument("--hidden_channels", type=int,   default=128)
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--batch_size",      type=int,   default=512)
    p.add_argument("--num_neighbors",   type=int,   default=20)
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--focal_alpha",     type=float, default=0.25)
    p.add_argument("--focal_gamma",     type=float, default=2.0)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    p.add_argument("--patience",        type=int,   default=15,
                   help="Early stopping patience in epochs (based on val AUC)")
    p.add_argument("--test_size",       type=float, default=0.2)
    p.add_argument("--val_size",        type=float, default=0.1)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()
    set_seed(SEED)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print(f"Device: {device}")

    # ── 1. Build graph ────────────────────────────────────────────────────────
    print("[1/4] Loading data and building static graph...")
    df                            = load_transactions(args.data)
    data, account_to_idx, _scaler = build_static_graph(df)

    n       = data.num_nodes
    n_mules = int(data.y.sum())
    print(f"      Nodes  : {n:,}")
    print(f"      Edges  : {data.num_edges:,}")
    print(f"      Mules  : {n_mules:,}  ({n_mules / n * 100:.2f}%)")
    print(f"      Feat   : {data.num_node_features}  (StandardScaler normalised)")

    # ── 2. Split nodes (train / val / test) ───────────────────────────────────
    indices                      = np.arange(n)
    trainval_idx, test_idx       = train_test_split(
        indices, test_size=args.test_size, random_state=SEED, stratify=data.y.numpy(),
    )
    val_frac = args.val_size / (1 - args.test_size)
    train_idx, val_idx           = train_test_split(
        trainval_idx, test_size=val_frac, random_state=SEED,
        stratify=data.y.numpy()[trainval_idx],
    )
    print(f"      Train : {len(train_idx):,} | Val : {len(val_idx):,} | Test : {len(test_idx):,}")

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx   = torch.tensor(val_idx,   dtype=torch.long)
    test_idx  = torch.tensor(test_idx,  dtype=torch.long)

    nb = [args.num_neighbors, args.num_neighbors]
    train_loader = NeighborLoader(data, num_neighbors=nb, batch_size=args.batch_size,
                                  input_nodes=train_idx, shuffle=True)
    val_loader   = NeighborLoader(data, num_neighbors=nb, batch_size=args.batch_size,
                                  input_nodes=val_idx,   shuffle=False)
    test_loader  = NeighborLoader(data, num_neighbors=nb, batch_size=args.batch_size,
                                  input_nodes=test_idx,  shuffle=False)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    print("[2/4] Initialising GraphSAGE...")
    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        dropout=args.dropout,
    ).to(device)
    print(f"      Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Decay LR by 0.5 when val AUC hasn't improved for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True,
    )

    # ── 4. Train with early stopping on val AUC ───────────────────────────────
    print("[3/4] Training...")
    best_val_auc   = 0.0
    best_state     = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss            = train_epoch(model, train_loader, optimizer,
                                            criterion, args.max_grad_norm)
        val_prob, val_true    = evaluate(model, val_loader)
        val_auc               = roc_auc_score(val_true, val_prob)
        scheduler.step(val_auc)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_auc > best_val_auc:
            best_val_auc       = val_auc
            best_state         = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve  = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no val AUC improvement for {args.patience} epochs)")
                break

    print(f"\n  Best val AUC: {best_val_auc:.4f} — restoring best weights")
    model.load_state_dict(best_state)

    # ── 5. Evaluate on held-out test set ──────────────────────────────────────
    print("[4/4] Evaluating on held-out test set...")
    y_prob, y_true = evaluate(model, test_loader)
    best_thresh    = find_best_threshold(y_true, y_prob)
    print(f"      Optimal threshold: {best_thresh:.4f}")

    metrics = compute_metrics(y_true, y_prob, threshold=best_thresh)
    print_metrics(metrics, model_name="GraphSAGE")
    print_confusion_matrix(y_true, y_prob, threshold=best_thresh)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    model_path   = os.path.join(OUTPUTS_DIR, "graphsage_model.pt")
    metrics_path = os.path.join(OUTPUTS_DIR, "graphsage_metrics.json")

    torch.save(model.state_dict(), model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel   → {model_path}")
    print(f"Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
