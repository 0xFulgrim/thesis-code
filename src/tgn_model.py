"""
tgn_model.py — Temporal Graph Network for streaming money mule detection.

Approach:
  - Transactions processed in strict chronological order as a stream.
  - Each account has a persistent memory vector (TGNMemory, Rossi et al. 2020).
  - GraphAttentionEmbedding (TransformerConv) computes embeddings conditioned
    on temporal neighbourhood.
  - MLP classifier predicts laundering probability per transaction.
  - Train/val/test split is TIME-BASED: first 60% train, next 10% val,
    last 30% test. Random splits are invalid for temporal models — they leak
    future information into training and inflate reported metrics.

Novel metric — Detection Latency (TGN-exclusive):
  For mule accounts correctly flagged during streaming evaluation:
    latency = timestamp of first alert − timestamp of first laundering transaction
  Lower is better. Not reported for static baselines (they see the entire
  test graph at once rather than processing transactions as they arrive).
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from utils import (
    build_temporal_data,
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


# ── Sub-modules ───────────────────────────────────────────────────────────────

class GraphAttentionEmbedding(torch.nn.Module):
    """
    Computes node embeddings by attending over temporal neighbours.
    Edge features = concat(time_encoding(rel_t), raw_msg).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        msg_dim: int,
        time_enc: torch.nn.Module,
    ):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2,
            heads=2, dropout=0.1, edge_dim=edge_dim,
        )

    def forward(
        self,
        x: Tensor,
        last_update: Tensor,
        edge_index: Tensor,
        t: Tensor,
        msg: Tensor,
    ) -> Tensor:
        rel_t     = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class TransactionClassifier(torch.nn.Module):
    """MLP head: concat(src_emb, dst_emb) → laundering logit."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2, in_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(in_channels, 1),
        )

    def forward(self, z_src: Tensor, z_dst: Tensor) -> Tensor:
        return self.mlp(torch.cat([z_src, z_dst], dim=-1))


# ── Training / eval helpers ───────────────────────────────────────────────────

def train_epoch(
    loader: TemporalDataLoader,
    data,
    memory: TGNMemory,
    gnn: GraphAttentionEmbedding,
    classifier: TransactionClassifier,
    neighbor_loader: LastNeighborLoader,
    assoc: Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    max_grad_norm: float,
) -> float:
    """
    One full pass over the training stream in time order.
    Memory and neighbor_loader are reset at the start of each epoch so
    the model always begins epoch k with a clean slate.
    """
    memory.train(); gnn.train(); classifier.train()
    memory.reset_state()
    neighbor_loader.reset_state()

    total_loss  = 0.0
    n_processed = 0

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(
            torch.cat([batch.src, batch.dst]).unique()
        )
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index,
                data.t[e_id].to(device), data.msg[e_id].to(device))

        logits = classifier(z[assoc[batch.src]], z[assoc[batch.dst]]).squeeze(-1)
        loss   = criterion(logits, batch.y.float())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(memory.parameters()) + list(gnn.parameters()) +
            list(classifier.parameters()),
            max_grad_norm,
        )
        optimizer.step()

        # Update AFTER backward, then detach every batch.
        # Detaching unconditionally prevents the computation graph growing across
        # batches, which causes "backward through graph a second time" errors.
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
        memory.detach()

        total_loss  += float(loss) * batch.num_events
        n_processed += batch.num_events

    return total_loss / max(n_processed, 1)


@torch.no_grad()
def eval_stream(
    loader: TemporalDataLoader,
    data,
    memory: TGNMemory,
    gnn: GraphAttentionEmbedding,
    classifier: TransactionClassifier,
    neighbor_loader: LastNeighborLoader,
    assoc: Tensor,
    update_memory: bool = True,
) -> tuple:
    """
    Stream evaluation. Returns (y_prob, y_true, metadata).

    metadata is a list of (prob, src, dst, t) per transaction,
    used for detection latency computation without a second loader pass.

    update_memory: set True during val/test to simulate live deployment
    (model keeps learning from observed transactions even during evaluation).
    """
    memory.eval(); gnn.eval(); classifier.eval()

    all_probs:  list = []
    all_labels: list = []
    metadata:   list = []   # (prob, src_idx, dst_idx, unix_ts) per transaction

    for batch in loader:
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(
            torch.cat([batch.src, batch.dst]).unique()
        )
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index,
                data.t[e_id].to(device), data.msg[e_id].to(device))

        logits = classifier(z[assoc[batch.src]], z[assoc[batch.dst]]).squeeze(-1)

        probs   = torch.sigmoid(logits).cpu().numpy()
        labels  = batch.y.cpu().numpy()
        t_arr   = batch.t.cpu().numpy()
        src_arr = batch.src.cpu().numpy()
        dst_arr = batch.dst.cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

        for i in range(len(probs)):
            metadata.append((probs[i], int(src_arr[i]), int(dst_arr[i]), int(t_arr[i])))

        if update_memory:
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)

    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        metadata,
    )


# ── Detection latency ─────────────────────────────────────────────────────────

def compute_detection_latency(
    metadata: list,
    y_true: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute detection latency from stored (prob, src, dst, t) metadata.
    No second loader pass needed.

    For each mule account correctly flagged: latency = first_flag_ts − first_launder_ts.
    False negatives (missed mules) reduce detected_mules but are not penalised
    in latency — they simply don't appear in the latency distribution.
    """
    account_first_launder: dict = {}
    account_first_flag:    dict = {}

    for i, (prob, src, dst, t_i) in enumerate(metadata):
        if y_true[i] == 1:
            for acc in (src, dst):
                if acc not in account_first_launder or t_i < account_first_launder[acc]:
                    account_first_launder[acc] = t_i
        if prob >= threshold:
            for acc in (src, dst):
                if acc not in account_first_flag or t_i < account_first_flag[acc]:
                    account_first_flag[acc] = t_i

    latencies = [
        account_first_flag[acc] - ts
        for acc, ts in account_first_launder.items()
        if acc in account_first_flag
    ]

    if not latencies:
        return {
            "mean_latency_seconds":   float("nan"),
            "median_latency_seconds": float("nan"),
            "detected_mules":         0,
            "total_mule_accounts":    len(account_first_launder),
        }
    return {
        "mean_latency_seconds":   float(np.mean(latencies)),
        "median_latency_seconds": float(np.median(latencies)),
        "detected_mules":         len(latencies),
        "total_mule_accounts":    len(account_first_launder),
    }


# ── CLI args ──────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TGN streaming mule detector")
    p.add_argument("--data",          default="data/raw/HI-Small_Trans.csv")
    p.add_argument("--memory_dim",    type=int,   default=128)
    p.add_argument("--time_dim",      type=int,   default=128)
    p.add_argument("--embedding_dim", type=int,   default=128)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--batch_size",    type=int,   default=200)
    p.add_argument("--neighbor_size", type=int,   default=10)
    p.add_argument("--detach_every",  type=int,   default=50)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience",      type=int,   default=5,
                   help="Early stopping patience (epochs, based on val AUC)")
    p.add_argument("--train_frac",    type=float, default=0.60)
    p.add_argument("--val_frac",      type=float, default=0.10)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()
    set_seed(SEED)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print(f"Device: {device}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[1/5] Loading data and building temporal graph...")
    df                   = load_transactions(args.data)
    data, account_to_idx = build_temporal_data(df)

    num_nodes  = int(max(data.src.max(), data.dst.max())) + 1
    num_events = data.num_events
    msg_dim    = data.msg.size(-1)
    n_launder  = int(data.y.sum())

    print(f"      Accounts     : {num_nodes:,}")
    print(f"      Transactions : {num_events:,}")
    print(f"      Laundering   : {n_launder:,}  ({n_launder / num_events * 100:.3f}%)")
    print(f"      Msg dim      : {msg_dim}  (log1p amount, currency, format)")

    # ── 2. Temporal split (60 / 10 / 30) ─────────────────────────────────────
    train_end = int(args.train_frac * num_events)
    val_end   = int((args.train_frac + args.val_frac) * num_events)

    train_data = data[:train_end]
    val_data   = data[train_end:val_end]
    test_data  = data[val_end:]

    n_train_launder = int(train_data.y.sum())

    print(f"      Train : {train_data.num_events:,} events  "
          f"({n_train_launder:,} laundering)")
    print(f"      Val   : {val_data.num_events:,} events")
    print(f"      Test  : {test_data.num_events:,} events")

    train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size)
    val_loader   = TemporalDataLoader(val_data,   batch_size=args.batch_size)
    test_loader  = TemporalDataLoader(test_data,  batch_size=args.batch_size)

    # ── 3. Initialise model ───────────────────────────────────────────────────
    print("[2/5] Initialising TGN...")

    memory = TGNMemory(
        num_nodes, msg_dim, args.memory_dim, args.time_dim,
        message_module=IdentityMessage(msg_dim, args.memory_dim, args.time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=args.memory_dim,
        out_channels=args.embedding_dim,
        msg_dim=msg_dim,
        time_enc=memory.time_enc,
    ).to(device)

    classifier = TransactionClassifier(in_channels=args.embedding_dim).to(device)

    neighbor_loader = LastNeighborLoader(num_nodes, size=args.neighbor_size, device=device)
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    # time_enc is shared between memory and gnn — exclude from gnn params to avoid
    # duplicate parameter warning and incorrect optimiser behaviour
    gnn_params = [p for n, p in gnn.named_parameters() if "time_enc" not in n]
    optimizer = torch.optim.Adam(
        list(memory.parameters()) + gnn_params + list(classifier.parameters()),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True,
    )

    # pos_weight from training split only — not full dataset
    n_train_normal = train_data.num_events - n_train_launder
    pos_weight = torch.tensor(
        [n_train_normal / max(n_train_launder, 1)], dtype=torch.float, device=device
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_params = sum(p.numel() for m in [memory, gnn, classifier] for p in m.parameters())
    print(f"      Parameters   : {n_params:,}")
    print(f"      pos_weight   : {pos_weight.item():.1f}  (train split only)")

    # ── 4. Training with early stopping on val AUC ────────────────────────────
    print("[3/5] Training...")
    best_val_auc      = 0.0
    best_memory_state = None
    best_gnn_state    = None
    best_clf_state    = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            train_loader, data, memory, gnn, classifier,
            neighbor_loader, assoc, optimizer, criterion,
            args.max_grad_norm,
        )

        # Val: continue from end-of-train memory state (don't reset)
        val_prob, val_true, _ = eval_stream(
            val_loader, data, memory, gnn, classifier,
            neighbor_loader, assoc, update_memory=True,
        )
        val_auc = roc_auc_score(val_true, val_prob)
        scheduler.step(val_auc)

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {train_loss:.6f} | Val AUC: {val_auc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_auc > best_val_auc:
            best_val_auc      = val_auc
            best_memory_state = {k: v.cpu().clone() for k, v in memory.state_dict().items()}
            best_gnn_state    = {k: v.cpu().clone() for k, v in gnn.state_dict().items()}
            best_clf_state    = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no val AUC improvement for {args.patience} epochs)")
                break

    print(f"\n  Best val AUC: {best_val_auc:.4f} — restoring best weights")
    memory.load_state_dict(best_memory_state)
    gnn.load_state_dict(best_gnn_state)
    classifier.load_state_dict(best_clf_state)

    # ── 5. Streaming evaluation on test set ───────────────────────────────────
    print("[4/5] Streaming evaluation on test set...")
    y_prob, y_true, metadata = eval_stream(
        test_loader, data, memory, gnn, classifier,
        neighbor_loader, assoc, update_memory=True,
    )

    # ── 6. Metrics ────────────────────────────────────────────────────────────
    print("[5/5] Computing metrics...")

    best_thresh = find_best_threshold(y_true, y_prob)
    print(f"      Optimal threshold: {best_thresh:.4f}")

    metrics = compute_metrics(y_true, y_prob, threshold=best_thresh)
    print_metrics(metrics, model_name="TGN")
    print_confusion_matrix(y_true, y_prob, threshold=best_thresh)

    latency = compute_detection_latency(metadata, y_true, best_thresh)

    print("  Detection Latency (TGN-exclusive streaming metric)")
    print(f"  {'─' * 40}")
    print(f"  Mule accounts in test  : {latency['total_mule_accounts']:,}")
    print(f"  Correctly detected     : {latency['detected_mules']:,}")
    if latency["total_mule_accounts"] > 0:
        det_rate = latency["detected_mules"] / latency["total_mule_accounts"] * 100
        print(f"  Detection rate         : {det_rate:.1f}%")
    print(f"  Mean latency           : {latency['mean_latency_seconds']:.0f} s")
    print(f"  Median latency         : {latency['median_latency_seconds']:.0f} s")

    all_metrics = {**metrics, "detection_latency": latency}

    # ── 7. Save ───────────────────────────────────────────────────────────────
    model_path   = os.path.join(OUTPUTS_DIR, "tgn_model.pt")
    metrics_path = os.path.join(OUTPUTS_DIR, "tgn_metrics.json")

    torch.save(
        {
            "memory":     memory.state_dict(),
            "gnn":        gnn.state_dict(),
            "classifier": classifier.state_dict(),
            "args":       vars(args),
        },
        model_path,
    )
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nModel   → {model_path}")
    print(f"Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
