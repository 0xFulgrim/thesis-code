"""
tgn_model.py — Temporal Graph Network for streaming detection of laundering transactions.

Approach:
  - Transactions are processed in strict chronological order as a stream.
  - Each account has a persistent memory vector (TGN, Rossi et al. 2020),
    updated by a GRU from the transactions the account takes part in.
  - A temporal attention layer (TransformerConv) embeds an account from its
    memory and its most recent counterparties.
  - An MLP scores each transaction from the embeddings of its two accounts,
    the transaction's own attributes, and how recently each account was active.
  - The split is TIME-BASED: first 60% train, next 10% validation, last 30%
    test. Random splits leak future information into training.

Evaluation:
  - The epoch with the best validation average precision is kept.
  - The decision threshold maximises F1 on the validation split. The test
    split is scored once, with that threshold.
  - Detection latency: for each mule account, the time from its first
    laundering transaction in the test stream to its first alert at or after it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time

import numpy as np
import torch
import torch_geometric
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import LastNeighborLoader

from data import DEFAULT_TRANS_PATH, Batch, Events, TransactionEncoder, load_events
from evaluation import (
    best_f1_threshold,
    classification_metrics,
    detection_latency,
    format_report,
)

# Settings that must not change between a run and its --resume.
RESUME_LOCKED = (
    "data", "max_events", "train_frac", "val_frac",
    "memory_dim", "time_dim", "embedding_dim", "neighbor_size", "no_txn_features",
    "batch_size", "lr", "pos_weight", "max_grad_norm", "seed",
)

_START = time.monotonic()


def log(message: str = "") -> None:
    """Print with the elapsed time, flushed so job logs update as the run goes."""
    elapsed = int(time.monotonic() - _START)
    stamp = f"{elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}"
    print(f"[{stamp}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Sub-modules ───────────────────────────────────────────────────────────────

class TimeEncoder(torch.nn.Module):
    """
    cos(w * dt + b) time encoding (Xu et al. 2020). The frequencies w start
    geometrically spaced from 1 down to 1e-9 rad/s, as in the TGN reference
    implementation, so the encoding resolves gaps from seconds to weeks.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = torch.nn.Linear(1, out_channels)
        with torch.no_grad():
            self.lin.weight.copy_(
                (10.0 ** -torch.linspace(0, 9, out_channels)).view(out_channels, 1)
            )
            self.lin.bias.zero_()

    def forward(self, dt: Tensor) -> Tensor:
        return self.lin(dt.to(self.lin.weight.dtype).view(-1, 1)).cos()


class StreamMemory(torch.nn.Module):
    """
    Per-account memory with a last-message store (Rossi et al. 2020).

    Every account has a memory vector and one pending message: its most
    recent transaction, which the memory has not absorbed yet. Reading an
    account applies its pending message through the GRU on the fly;
    update_state writes that result back for the accounts of a batch and
    makes their latest transaction in the batch the new pending message.
    Because a transaction only becomes pending after its batch was scored,
    a batch never influences its own predictions.

    The message of an account is
      [own memory, counterparty memory, transaction features,
       whether it paid or received, time since its previous transaction].

    The store is a set of dense per-account tensors, so every operation is
    vectorised, and the module behaves identically in train and eval mode.
    All of the stream state is in the state dict.
    """

    def __init__(self, num_nodes: int, msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.time_enc = TimeEncoder(time_dim)
        self.gru = torch.nn.GRUCell(2 * memory_dim + msg_dim + 1 + time_dim, memory_dim)

        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes, dtype=torch.long))
        # Pending message of every account.
        self.register_buffer("pending", torch.zeros(num_nodes, dtype=torch.bool))
        self.register_buffer("pending_other", torch.zeros(num_nodes, dtype=torch.long))
        self.register_buffer("pending_t", torch.zeros(num_nodes, dtype=torch.long))
        self.register_buffer("pending_msg", torch.zeros(num_nodes, msg_dim))
        self.register_buffer("pending_paid", torch.zeros(num_nodes, dtype=torch.bool))

    def reset_state(self) -> None:
        for buffer in self.buffers():
            buffer.zero_()

    def forward(self, n_id: Tensor) -> tuple[Tensor, Tensor]:
        """Memory and time of the latest transaction of the given accounts."""
        memory = self.memory[n_id]
        last_update = self.last_update[n_id]

        has_pending = self.pending[n_id]
        node = n_id[has_pending]
        if node.numel() > 0:
            t = self.pending_t[node]
            previous = self.last_update[node]
            # An account's first transaction has no previous one: use a gap of 0.
            gap = torch.where(previous > 0, t - previous, torch.zeros_like(t))
            message = torch.cat([
                self.memory[node],
                self.memory[self.pending_other[node]],
                self.pending_msg[node],
                self.pending_paid[node].to(memory.dtype).unsqueeze(-1),
                self.time_enc(gap),
            ], dim=-1)
            memory[has_pending] = self.gru(message, self.memory[node])
            last_update[has_pending] = t
        return memory, last_update

    @torch.no_grad()
    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor) -> None:
        """Let a scored batch of transactions (in stream order) into the memory."""
        n_id, inverse = torch.cat([src, dst]).unique(return_inverse=True)

        # Absorb what was pending for these accounts...
        memory, last_update = self(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

        # ...and make each account's latest transaction of the batch pending.
        # Entry j is transaction j as seen by its payer, entry b + j the same
        # transaction as seen by its receiver. The later transaction wins; if
        # an account pays itself, the payer's view wins.
        b = src.numel()
        position = torch.arange(b, device=src.device).repeat(2)
        is_payer = torch.arange(2 * b, device=src.device) < b
        key = position * 2 + is_payer
        latest = key.new_full((n_id.numel(),), -1)
        latest.scatter_reduce_(0, inverse, key, reduce="amax", include_self=True)
        j, paid = latest // 2, latest % 2 == 1

        self.pending[n_id] = True
        self.pending_other[n_id] = torch.where(paid, dst[j], src[j])
        self.pending_t[n_id] = t[j]
        self.pending_msg[n_id] = msg[j]
        self.pending_paid[n_id] = paid


class GraphAttentionEmbedding(torch.nn.Module):
    """
    Computes node embeddings by attending over temporal neighbours.
    Edge features = concat(time_encoding(rel_t), transaction features).
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
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class TransactionClassifier(torch.nn.Module):
    """MLP head: concat(src_emb, dst_emb, transaction features) → laundering logit."""

    def __init__(self, in_channels: int, extra_channels: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2 + extra_channels, in_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(in_channels, 1),
        )

    def forward(self, z_src: Tensor, z_dst: Tensor, extra: Tensor | None = None) -> Tensor:
        parts = [z_src, z_dst] if extra is None else [z_src, z_dst, extra]
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)


def recency(t: Tensor, last_update: Tensor) -> Tensor:
    """
    Two features per account: the log of the seconds since its previous
    transaction (scaled to stay of order one), and whether this is the first
    time the account is seen. last_update is 0 for accounts not seen before.
    """
    seen = last_update > 0
    elapsed = (t - last_update).clamp(min=0).to(torch.float32)
    return torch.stack([
        torch.log1p(elapsed) * seen / 10.0,
        (~seen).to(torch.float32),
    ], dim=-1)


# ── Model ─────────────────────────────────────────────────────────────────────

class TGN(torch.nn.Module):
    """
    The memory, the temporal embedding and the classifier, together with the
    stream state they share: per-account memory and each account's most recent
    neighbours.
    """

    def __init__(
        self,
        num_nodes: int,
        encoder: TransactionEncoder,
        memory_dim: int,
        time_dim: int,
        embedding_dim: int,
        neighbor_size: int,
        use_txn_features: bool,
        device: torch.device,
    ):
        super().__init__()
        msg_dim = encoder.out_channels
        self.encoder = encoder
        self.use_txn_features = use_txn_features

        self.memory = StreamMemory(num_nodes, msg_dim, memory_dim, time_dim)
        # The time encoder is shared between the memory and the embedding.
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=msg_dim,
            time_enc=self.memory.time_enc,
        )
        # Transaction features, plus two recency features for each account.
        extra = msg_dim + 4 if use_txn_features else 0
        self.classifier = TransactionClassifier(embedding_dim, extra)

        self.register_buffer("assoc", torch.empty(num_nodes, dtype=torch.long), persistent=False)
        self.to(device)
        # Not a module: it keeps its tensors on the device it is created with.
        self.neighbors = LastNeighborLoader(num_nodes, size=neighbor_size, device=device)
        self.reset_state()

    @property
    def device(self) -> torch.device:
        return self.assoc.device

    def reset_state(self) -> None:
        """Forget the stream: clear every account's memory and neighbours."""
        self.memory.reset_state()
        self.neighbors.reset_state()

    def forward(self, batch: Batch, events: Events) -> tuple[Tensor, Tensor]:
        """
        Laundering logits for a batch, computed from the state as it was
        before the batch. Also returns the batch's feature vectors, which
        update_state needs.
        """
        msg = self.encoder(batch)

        n_id, edge_index, e_id = self.neighbors(torch.cat([batch.src, batch.dst]).unique())
        self.assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        memory, last_update = self.memory(n_id)
        past = events.rows(e_id, n_id.device)
        z = self.gnn(memory, last_update, edge_index, past.t, self.encoder(past))

        src, dst = self.assoc[batch.src], self.assoc[batch.dst]
        extra = None
        if self.use_txn_features:
            extra = torch.cat([
                msg,
                recency(batch.t, last_update[src]),
                recency(batch.t, last_update[dst]),
            ], dim=-1)
        return self.classifier(z[src], z[dst], extra), msg

    def update_state(self, batch: Batch, msg: Tensor) -> None:
        """Let the batch's transactions into the memory and the neighbour lists."""
        self.memory.update_state(batch.src, batch.dst, batch.t, msg)
        self.neighbors.insert(batch.src, batch.dst)


# ── Training / eval helpers ───────────────────────────────────────────────────

class Progress:
    """Logs throughput at most once a minute, so long passes show signs of life."""

    def __init__(self, label: str, total: int):
        self.label = label
        self.total = total
        self.start = self.last = time.monotonic()

    def update(self, done: int) -> None:
        now = time.monotonic()
        if now - self.last >= 60:
            self.last = now
            rate = done / (now - self.start)
            log(f"      {self.label}: {done / self.total:4.0%}  ({rate:,.0f} transactions/s)")


def train_epoch(
    model: TGN,
    events: Events,
    start: int,
    end: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    max_grad_norm: float,
) -> float:
    """
    One pass over the training stream in time order. The stream state is
    reset first, so every epoch begins with a clean slate.
    """
    model.train()
    model.reset_state()

    total_loss = 0.0
    progress = Progress("train", end - start)

    for lo in range(start, end, batch_size):
        hi = min(lo + batch_size, end)
        batch = events.rows(slice(lo, hi), model.device)

        optimizer.zero_grad()
        logits, msg = model(batch, events)
        loss = criterion(logits, batch.y.to(logits.dtype))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Update AFTER the batch was scored and the weights were stepped.
        model.update_state(batch, msg)

        total_loss += loss.detach().item() * (hi - lo)
        progress.update(hi - start)

    return total_loss / max(end - start, 1)


@torch.no_grad()
def score_stream(model: TGN, events: Events, start: int, end: int, batch_size: int) -> np.ndarray:
    """
    Laundering probability of every transaction in events[start:end], in
    stream order. Each batch is scored before the model sees it, then let
    into the stream state, as it would be in live use.
    """
    model.eval()
    scores = torch.empty(end - start, dtype=torch.float64)
    progress = Progress("score", end - start)

    for lo in range(start, end, batch_size):
        hi = min(lo + batch_size, end)
        batch = events.rows(slice(lo, hi), model.device)
        logits, msg = model(batch, events)
        # float64: float32 sigmoids saturate at 1.0 and tie the top scores.
        scores[lo - start:hi - start] = torch.sigmoid(logits.double()).cpu()
        model.update_state(batch, msg)
        progress.update(hi - start)

    return scores.numpy()


def replay_neighbors(model: TGN, events: Events, segments: list, batch_size: int) -> None:
    """
    Rebuild the neighbour lists as they are after streaming the given
    (start, end) segments. They depend only on the data and the batch
    boundaries, not on the model weights.
    """
    model.neighbors.reset_state()
    for start, end in segments:
        for lo in range(start, end, batch_size):
            hi = min(lo + batch_size, end)
            model.neighbors.insert(
                events.src[lo:hi].to(model.device), events.dst[lo:hi].to(model.device),
            )


def save_atomic(obj: dict, path: str) -> None:
    """Write via a temporary file, so an interrupted save never corrupts `path`."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


# ── CLI args ──────────────────────────────────────────────────────────────────

def get_args(argv: list | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TGN streaming laundering detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("data")
    g.add_argument("--data", default=DEFAULT_TRANS_PATH,
                   help="A *_Trans.csv file from the IBM AML dataset")
    g.add_argument("--max_events", type=int, default=None,
                   help="Use only the first N transactions (quick end-to-end check)")
    g.add_argument("--no_cache", action="store_true",
                   help="Do not read or write the parsed-events cache next to the CSV")
    g.add_argument("--train_frac", type=float, default=0.60)
    g.add_argument("--val_frac", type=float, default=0.10)

    g = p.add_argument_group("model")
    g.add_argument("--memory_dim", type=int, default=128)
    g.add_argument("--time_dim", type=int, default=128)
    g.add_argument("--embedding_dim", type=int, default=128)
    g.add_argument("--neighbor_size", type=int, default=10)
    g.add_argument("--no_txn_features", action="store_true",
                   help="Ablation: classify from the two account embeddings alone")

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--batch_size", type=int, default=200)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--pos_weight", type=float, default=None,
                   help="Loss weight of laundering transactions "
                        "(default: negatives / positives in the training split)")
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience (epochs, based on val average precision)")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("run")
    g.add_argument("--device", default=None, help="cuda, cuda:1, cpu ... (default: cuda if available)")
    g.add_argument("--out_dir", default="outputs")
    g.add_argument("--resume", action="store_true",
                   help="Continue from the checkpoint in --out_dir, if there is one")
    g.add_argument("--save_scores", action="store_true",
                   help="Also save the test scores, for plots without re-running")
    return p.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list | None = None) -> dict:
    args = get_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "tgn_model.pt")
    checkpoint_path = os.path.join(args.out_dir, "tgn_checkpoint.pt")
    metrics_path = os.path.join(args.out_dir, "tgn_metrics.json")
    scores_path = os.path.join(args.out_dir, "tgn_test_scores.npz")

    log(f"Device: {device}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log("[1/5] Loading transactions...")
    events = load_events(args.data, use_cache=not args.no_cache, log=log)
    if args.max_events is not None:
        events = events.head(args.max_events)

    num_events = events.num_events
    labels = events.y.numpy()
    n_launder = int(labels.sum())

    log(f"      Accounts     : {events.num_nodes:,}")
    log(f"      Transactions : {num_events:,}")
    log(f"      Laundering   : {n_launder:,}  ({n_launder / num_events * 100:.3f}%)")

    # ── 2. Temporal split ─────────────────────────────────────────────────────
    train_end = int(args.train_frac * num_events)
    val_end = int((args.train_frac + args.val_frac) * num_events)
    splits = {"train": (0, train_end), "validation": (train_end, val_end), "test": (val_end, num_events)}
    split_stats = {}
    for name, (lo, hi) in splits.items():
        split_stats[name] = {"transactions": hi - lo, "laundering": int(labels[lo:hi].sum())}
        log(f"      {name.capitalize():<11}: {hi - lo:,} transactions  "
            f"({split_stats[name]['laundering']:,} laundering)")

    for name in ("train", "validation"):
        stats = split_stats[name]
        if stats["laundering"] in (0, stats["transactions"]):
            raise SystemExit(
                f"The {name} split needs both laundering and normal transactions; "
                "use more data or different split fractions."
            )
    if split_stats["test"]["transactions"] == 0:
        raise SystemExit("The test split is empty; lower --train_frac or --val_frac.")

    y_val = labels[train_end:val_end]
    y_test = labels[val_end:]

    # ── 3. Initialise model ───────────────────────────────────────────────────
    log("[2/5] Initialising TGN...")
    model = TGN(
        num_nodes=events.num_nodes,
        encoder=TransactionEncoder.fit(events, train_end),
        memory_dim=args.memory_dim,
        time_dim=args.time_dim,
        embedding_dim=args.embedding_dim,
        neighbor_size=args.neighbor_size,
        use_txn_features=not args.no_txn_features,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    # pos_weight from the training split only — not the full dataset
    n_train_launder = split_stats["train"]["laundering"]
    pos_weight = args.pos_weight
    if pos_weight is None:
        pos_weight = (train_end - n_train_launder) / n_train_launder
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float, device=device),
    )

    log(f"      Parameters   : {sum(p.numel() for p in model.parameters()):,}")
    log(f"      Feature dim  : {model.encoder.out_channels}")
    log(f"      pos_weight   : {pos_weight:.1f}")

    # ── 4. Training with early stopping on val average precision ──────────────
    history: list = []
    best_ap = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    first_epoch = 1

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key in RESUME_LOCKED:
            if checkpoint["args"][key] != getattr(args, key):
                raise SystemExit(
                    f"--resume: --{key} was {checkpoint['args'][key]!r} in the "
                    f"checkpointed run but is {getattr(args, key)!r} now."
                )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        if checkpoint["rng"]["cuda"] and device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        history = checkpoint["history"]
        best_ap = checkpoint["best_ap"]
        best_epoch = checkpoint["best_epoch"]
        epochs_no_improve = checkpoint["epochs_no_improve"]
        first_epoch = len(history) + 1
        log(f"      Resumed after epoch {len(history)} (best val AP {best_ap:.4f})")

    log("[3/5] Training...")
    for epoch in range(first_epoch, args.epochs + 1):
        if epochs_no_improve >= args.patience:
            break
        epoch_start = time.monotonic()

        train_loss = train_epoch(
            model, events, 0, train_end, args.batch_size,
            optimizer, criterion, args.max_grad_norm,
        )
        # Val: continue from the end-of-train stream state (don't reset)
        val_prob = score_stream(model, events, train_end, val_end, args.batch_size)
        val_ap = float(average_precision_score(y_val, val_prob))
        val_auc = float(roc_auc_score(y_val, val_prob))
        scheduler.step(val_ap)

        lr = optimizer.param_groups[0]["lr"]
        seconds = time.monotonic() - epoch_start
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_avg_precision": val_ap, "val_auc_roc": val_auc,
            "lr": lr, "seconds": seconds,
        })
        log(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.6f} | "
            f"Val AP: {val_ap:.4f} | Val AUC: {val_auc:.4f} | "
            f"LR: {lr:.2e} | {seconds:,.0f} s")

        if val_ap > best_ap:
            best_ap, best_epoch, epochs_no_improve = val_ap, epoch, 0
            threshold = best_f1_threshold(y_val, val_prob)
            # The saved state includes every account's memory and pending
            # message as they stand after the validation stream, which is
            # where the test stream starts.
            save_atomic({
                "model": model.state_dict(),
                "epoch": epoch,
                "threshold": threshold,
                "validation": classification_metrics(y_val, val_prob, threshold),
                "currencies": events.currencies,
                "formats": events.formats,
                "args": vars(args),
            }, model_path)
        else:
            epochs_no_improve += 1

        save_atomic({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
            "history": history,
            "best_ap": best_ap,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "args": vars(args),
        }, checkpoint_path)

        if epochs_no_improve >= args.patience:
            log(f"  Early stopping at epoch {epoch} "
                f"(no val AP improvement for {args.patience} epochs)")

    if not history:
        raise SystemExit("Nothing was trained: --epochs must be at least 1.")

    # ── 5. Streaming evaluation on test set ───────────────────────────────────
    log(f"[4/5] Restoring epoch {best_epoch} (val AP {best_ap:.4f}) and scoring the test stream...")
    best = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    replay_neighbors(model, events, [splits["train"], splits["validation"]], args.batch_size)

    test_prob = score_stream(model, events, val_end, num_events, args.batch_size)

    # ── 6. Metrics ────────────────────────────────────────────────────────────
    log("[5/5] Computing metrics...")
    threshold = best["threshold"]
    test_metrics = classification_metrics(y_test, test_prob, threshold)
    latency = detection_latency(
        events.src[val_end:].numpy(), events.dst[val_end:].numpy(),
        events.t[val_end:].numpy(), y_test, test_prob >= threshold,
    )
    print(format_report("TGN — validation (threshold selected here)", best["validation"]))
    print(format_report("TGN — test", test_metrics, latency), flush=True)

    results = {
        "data": {
            "file": args.data,
            "accounts": events.num_nodes,
            "transactions": num_events,
            "laundering": n_launder,
            "splits": split_stats,
        },
        "best_epoch": best_epoch,
        "validation": best["validation"],
        "test": test_metrics,
        "detection_latency": latency,
        "history": history,
        "train_seconds": sum(h["seconds"] for h in history),
        "config": {**vars(args), "pos_weight_used": pos_weight, "device_used": str(device)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }

    # ── 7. Save ───────────────────────────────────────────────────────────────
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Metrics → {metrics_path}")
    log(f"Model   → {model_path}")

    if args.save_scores:
        np.savez_compressed(
            scores_path,
            score=test_prob, label=y_test,
            src=events.src[val_end:].numpy().astype(np.int32),
            dst=events.dst[val_end:].numpy().astype(np.int32),
            t=events.t[val_end:].numpy(),
            threshold=threshold,
        )
        log(f"Scores  → {scores_path}")

    return results


if __name__ == "__main__":
    main()
