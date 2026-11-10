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
  - The test metrics are also reported without the quiet days at the end of
    the files, where almost every remaining transaction is laundering.

Training settings: the loss is unweighted and gradients are not clipped by
default. On HI-Small, a class weight of about 1,300 combined with clipping at
1.0 scaled the gradients of batches that contain laundering down by a factor
of 100 to 10,000, and the model stalled (validation AP 0.33 against 0.53).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
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
    quiet_day_mask,
)

# Settings that must not change between a run and its --resume.
RESUME_LOCKED = (
    "data", "max_events", "train_frac", "val_frac",
    "memory_dim", "time_dim", "embedding_dim", "neighbor_size", "layers", "hop2_size",
    "no_txn_features", "no_activity_features", "activity_in_attention",
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


class AccountActivity(torch.nn.Module):
    """
    Running activity statistics of every account, kept as the stream goes by:

      payments made and received so far (log counts)                      2
      the same, decayed with half-lives of an hour, a day and a week      6
      mean standardised log amount paid and received                      2
      time since the account was first seen (log)                         1

    They give the classifier counts that the memory would otherwise have to
    learn to keep. Like the memory, they only learn about a batch after it
    was scored. No trainable parameters.
    """

    HALF_LIVES = (3_600.0, 86_400.0, 604_800.0)
    out_channels = 11

    def __init__(self, num_nodes: int):
        super().__init__()
        k = len(self.HALF_LIVES)
        self.register_buffer("half_life", torch.tensor(self.HALF_LIVES))
        # Index 0 of the second dimension: as payer. Index 1: as receiver.
        self.register_buffer("count", torch.zeros(num_nodes, 2))
        self.register_buffer("amount_sum", torch.zeros(num_nodes, 2))
        self.register_buffer("recent", torch.zeros(num_nodes, 2, k))      # as of `recent_t`
        self.register_buffer("recent_t", torch.zeros(num_nodes, dtype=torch.long))
        self.register_buffer("first_seen", torch.zeros(num_nodes, dtype=torch.long))

    def reset_state(self) -> None:
        for buffer in (self.count, self.amount_sum, self.recent, self.recent_t, self.first_seen):
            buffer.zero_()

    def forward(self, node: Tensor, t: Tensor) -> Tensor:
        """The statistics of `node` as they stand at time `t`."""
        count = self.count[node]
        elapsed = (t - self.recent_t[node]).clamp(min=0).to(count.dtype)
        decay = torch.pow(0.5, elapsed.unsqueeze(-1) / self.half_life)
        recent = self.recent[node] * decay.unsqueeze(1)
        first_seen = self.first_seen[node]
        age = (t - first_seen).clamp(min=0).to(count.dtype) * (first_seen > 0)
        return torch.cat([
            torch.log1p(count) / 5,
            torch.log1p(recent).flatten(1) / 5,
            self.amount_sum[node] / count.clamp(min=1),
            torch.log1p(age).unsqueeze(-1) / 10,
        ], dim=-1)

    @torch.no_grad()
    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor) -> None:
        """Count a scored batch. msg[:, 0] and msg[:, 1] are the standardised log amounts."""
        b, k = src.numel(), self.half_life.numel()
        # Entry j is transaction j for its payer, entry b + j for its receiver.
        node = torch.cat([src, dst])
        role = (torch.arange(2 * b, device=src.device) >= b).long()
        time = t.repeat(2)
        n_id, inverse = node.unique(return_inverse=True)

        self.count.view(-1).index_add_(0, node * 2 + role, torch.ones_like(time, dtype=self.count.dtype))
        self.amount_sum.view(-1).index_add_(0, node * 2 + role, torch.cat([msg[:, 0], msg[:, 1]]))

        # Bring each account's decayed counts forward to its latest transaction
        # in the batch, then add the batch's transactions, decayed to that time.
        latest = time.new_zeros(n_id.numel())
        latest.scatter_reduce_(0, inverse, time, reduce="amax", include_self=True)
        elapsed = (latest - self.recent_t[n_id]).clamp(min=0).to(self.recent.dtype)
        carried = self.recent[n_id] * torch.pow(0.5, elapsed.unsqueeze(-1) / self.half_life).unsqueeze(1)
        since = (latest[inverse] - time).to(self.recent.dtype)
        added = self.recent.new_zeros(n_id.numel() * 2, k)
        added.index_add_(0, inverse * 2 + role, torch.pow(0.5, since.unsqueeze(-1) / self.half_life))
        self.recent[n_id] = carried + added.view(-1, 2, k)
        self.recent_t[n_id] = latest

        earliest = time.new_full((n_id.numel(),), torch.iinfo(torch.long).max)
        earliest.scatter_reduce_(0, inverse, time, reduce="amin", include_self=True)
        new = self.first_seen[n_id] == 0
        self.first_seen[n_id[new]] = earliest[new]


class GraphAttentionEmbedding(torch.nn.Module):
    """
    Computes node embeddings by attending over temporal neighbours, once per
    layer. Edge features = concat(time_encoding(rel_t), transaction features).
    With two layers an account also sees its neighbours' neighbours.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        msg_dim: int,
        time_enc: torch.nn.Module,
        num_layers: int = 1,
    ):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.convs = torch.nn.ModuleList(
            TransformerConv(
                in_channels if layer == 0 else out_channels, out_channels // 2,
                heads=2, dropout=0.1, edge_dim=edge_dim,
            )
            for layer in range(num_layers)
        )

    def forward(
        self,
        x: Tensor,
        last_update: Tensor,
        edge_index: Tensor,
        t: Tensor,
        msg: Tensor,
        num_seed_edges: int | None = None,
    ) -> Tensor:
        """
        The first num_seed_edges edges are the ones into the accounts whose
        embedding is wanted. The last layer only runs on those: the other
        edges cannot change the embeddings that are read, so skipping them
        gives the same result for a fraction of the work.
        """
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        for layer, conv in enumerate(self.convs):
            if layer > 0:
                x = x.relu()
            if layer == len(self.convs) - 1 and num_seed_edges is not None:
                x = conv(x, edge_index[:, :num_seed_edges], edge_attr[:num_seed_edges])
            else:
                x = conv(x, edge_index, edge_attr)
        return x


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
        num_layers: int = 1,
        use_activity_features: bool = True,
        activity_in_attention: bool = False,
        hop2_size: int | None = None,
    ):
        super().__init__()
        if num_layers not in (1, 2):
            raise ValueError("num_layers must be 1 or 2")
        self.neighbor_size = neighbor_size
        self.hop2_size = min(hop2_size or neighbor_size, neighbor_size)
        msg_dim = encoder.out_channels
        self.encoder = encoder
        self.use_txn_features = use_txn_features
        self.use_activity_features = use_activity_features
        self.num_layers = num_layers

        self.memory = StreamMemory(num_nodes, msg_dim, memory_dim, time_dim)
        self.activity = AccountActivity(num_nodes)
        # The time encoder is shared between the memory and the embedding.
        # Past transactions carry one more feature there: which way the money went.
        self.activity_in_attention = activity_in_attention
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim + (AccountActivity.out_channels if activity_in_attention else 0),
            out_channels=embedding_dim,
            msg_dim=msg_dim + 1,
            time_enc=self.memory.time_enc,
            num_layers=num_layers,
        )
        extra = 0
        if use_txn_features:
            extra += msg_dim + 4          # the transaction, and two recency features per account
        if use_activity_features:
            extra += 2 * AccountActivity.out_channels + 10   # both accounts' activity, and the local structure
        self.classifier = TransactionClassifier(embedding_dim, extra)

        self.register_buffer("assoc", torch.empty(num_nodes, dtype=torch.long), persistent=False)
        self.register_buffer("is_seed", torch.zeros(num_nodes, dtype=torch.bool), persistent=False)
        self.to(device)
        # Not a module: it keeps its tensors on the device it is created with.
        self.neighbors = LastNeighborLoader(num_nodes, size=neighbor_size, device=device)
        self.reset_state()

    @property
    def device(self) -> torch.device:
        return self.assoc.device

    def reset_state(self) -> None:
        """Forget the stream: clear every account's memory, statistics and neighbours."""
        self.memory.reset_state()
        self.activity.reset_state()
        self.neighbors.reset_state()

    def local_structure(self, batch: Batch, events: Events) -> Tensor:
        """
        What the two accounts' most recent transactions say about the payment
        pattern this transaction is part of (log counts):

          earlier payments from payer to receiver, and back                  2
          third accounts X both dealt with, by who paid whom:
            payer paid X, X paid receiver   (a second route for the money)
            receiver paid X, X paid payer   (this payment closes a loop)
            both paid X                     (they fan in to X)
            X paid both                     (X fans out to them)              4
          different accounts the payer recently paid / was paid by           2
          the same for the receiver                                          2
        """
        device = batch.src.device
        src, dst = batch.src.unsqueeze(-1), batch.dst.unsqueeze(-1)

        def recent(account: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            e_id = self.neighbors.e_id[account]                     # [B, size]; -1 marks an empty slot
            partner = self.neighbors.neighbors[account]
            payer = events.src[e_id.clamp(min=0).to(events.src.device)].to(device)
            return e_id >= 0, partner, payer == account.unsqueeze(-1)

        def distinct(partner: Tensor, mask: Tensor) -> Tensor:
            ordered = torch.where(mask, partner, torch.full_like(partner, -1)).sort(dim=-1).values
            changes = (ordered[:, 1:] != ordered[:, :-1]) & (ordered[:, 1:] >= 0)
            return changes.sum(-1) + (ordered[:, 0] >= 0)

        s_valid, s_partner, s_paid = recent(batch.src)
        d_valid, d_partner, d_paid = recent(batch.dst)

        together = s_valid & (s_partner == dst)
        s_third = s_valid & (s_partner != src) & (s_partner != dst)
        d_third = d_valid & (d_partner != src) & (d_partner != dst)
        shared = (s_partner.unsqueeze(2) == d_partner.unsqueeze(1)) & s_third.unsqueeze(2) & d_third.unsqueeze(1)
        sp, dp = s_paid.unsqueeze(2), d_paid.unsqueeze(1)

        counts = torch.stack([
            (together & s_paid).sum(-1),
            (together & ~s_paid).sum(-1),
            (shared & sp & ~dp).sum((1, 2)),
            (shared & ~sp & dp).sum((1, 2)),
            (shared & sp & dp).sum((1, 2)),
            (shared & ~sp & ~dp).sum((1, 2)),
            distinct(s_partner, s_valid & s_paid),
            distinct(s_partner, s_valid & ~s_paid),
            distinct(d_partner, d_valid & d_paid),
            distinct(d_partner, d_valid & ~d_paid),
        ], dim=-1)
        return torch.log1p(counts.to(torch.float32))

    def recent_edges(self, accounts: Tensor, size: int) -> tuple[Tensor, Tensor, Tensor]:
        """
        The `size` most recent transactions of each account, as edges from
        the counterparty to the account: (counterparty, account, transaction id).
        """
        e_id = self.neighbors.e_id[accounts, :size]             # most recent first; -1 = empty slot
        counterparty = self.neighbors.neighbors[accounts, :size]
        filled = e_id >= 0
        return counterparty[filled], accounts.unsqueeze(-1).expand_as(e_id)[filled], e_id[filled]

    def neighborhood(self, seeds: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
        """
        What the attention layers need for the given accounts: their recent
        transactions and, with a second layer, the recent transactions of the
        accounts those involve. Returns the accounts involved, the edges
        (indexing into them), the transaction ids, and how many of the edges,
        the first ones, lead into the seeds themselves.
        """
        source, target, e_id = self.recent_edges(seeds, self.neighbor_size)
        num_seed_edges = e_id.numel()
        if self.num_layers == 2:
            self.is_seed[seeds] = True
            ring = source.unique()
            ring = ring[~self.is_seed[ring]]                     # one hop out, not seeds themselves
            self.is_seed[seeds] = False
            ring_source, ring_target, ring_e_id = self.recent_edges(ring, self.hop2_size)
            source = torch.cat([source, ring_source])
            target = torch.cat([target, ring_target])
            e_id = torch.cat([e_id, ring_e_id])

        n_id = torch.cat([seeds, source]).unique()
        self.assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)
        return n_id, torch.stack([self.assoc[source], self.assoc[target]]), e_id, num_seed_edges

    def forward(self, batch: Batch, events: Events) -> tuple[Tensor, Tensor]:
        """
        Laundering logits for a batch, computed from the state as it was
        before the batch. Also returns the batch's feature vectors, which
        update_state needs.
        """
        msg = self.encoder(batch)

        n_id, edge_index, e_id, num_seed_edges = self.neighborhood(
            torch.cat([batch.src, batch.dst]).unique()
        )
        memory, last_update = self.memory(n_id)
        past = events.rows(e_id, n_id.device)
        # Each past transaction is attended to by one of its two accounts:
        # tell the layer whether that account was the payer.
        was_payer = (past.src == n_id[edge_index[1]]).to(msg.dtype).unsqueeze(-1)
        x = memory
        if self.activity_in_attention:
            # Every account's statistics as they stand at the end of the batch's time span.
            x = torch.cat([memory, self.activity(n_id, batch.t.max().expand(n_id.numel()))], dim=-1)
        z = self.gnn(x, last_update, edge_index, past.t,
                     torch.cat([self.encoder(past), was_payer], dim=-1), num_seed_edges)

        src, dst = self.assoc[batch.src], self.assoc[batch.dst]
        extra = []
        if self.use_txn_features:
            extra += [msg, recency(batch.t, last_update[src]), recency(batch.t, last_update[dst])]
        if self.use_activity_features:
            extra += [
                self.activity(batch.src, batch.t),
                self.activity(batch.dst, batch.t),
                self.local_structure(batch, events),
            ]
        return self.classifier(z[src], z[dst], torch.cat(extra, dim=-1) if extra else None), msg

    def update_state(self, batch: Batch, msg: Tensor) -> None:
        """Let the batch's transactions into the memory, the statistics and the neighbour lists."""
        self.memory.update_state(batch.src, batch.dst, batch.t, msg)
        self.activity.update_state(batch.src, batch.dst, batch.t, msg)
        self.neighbors.insert(batch.src, batch.dst)


# ── Training / eval helpers ───────────────────────────────────────────────────

class Progress:
    """Logs throughput and time left at most once a minute, so long passes show signs of life."""

    def __init__(self, label: str, total: int):
        self.label = label
        self.total = total
        self.start = self.last = time.monotonic()

    def due(self) -> bool:
        return time.monotonic() - self.last >= 60

    def report(self, done: int, extra: str = "") -> None:
        self.last = time.monotonic()
        rate = done / (self.last - self.start)
        minutes_left = (self.total - done) / rate / 60
        log(f"      {self.label}: {done / self.total:4.0%}  ({rate:,.0f} transactions/s, "
            f"about {minutes_left:,.0f} min left{extra})")


def mean_loss(loss_sum: Tensor, count: int) -> float:
    """Mean loss so far. Stops the run if it is no longer a finite number."""
    mean = loss_sum.item() / max(count, 1)
    if not math.isfinite(mean):
        raise FloatingPointError(
            "The training loss is no longer finite. The last finished epoch is "
            "still checkpointed; try a lower --lr, or set --max_grad_norm."
        )
    return mean


def train_epoch(
    model: TGN,
    events: Events,
    start: int,
    end: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    max_grad_norm: float | None,
) -> float:
    """
    One pass over the training stream in time order. The stream state is
    reset first, so every epoch begins with a clean slate.
    """
    model.train()
    model.reset_state()

    # Summed on the device and read about once a minute: reading the loss of
    # every batch would make the GPU wait for the Python loop each time.
    loss_sum = torch.zeros((), dtype=torch.float64, device=model.device)
    progress = Progress("train", end - start)

    for lo in range(start, end, batch_size):
        hi = min(lo + batch_size, end)
        batch = events.rows(slice(lo, hi), model.device)

        optimizer.zero_grad()
        logits, msg = model(batch, events)
        loss = criterion(logits, batch.y.to(logits.dtype))
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Update AFTER the batch was scored and the weights were stepped.
        model.update_state(batch, msg)

        loss_sum += loss.detach() * (hi - lo)
        if progress.due():
            progress.report(hi - start, f", loss {mean_loss(loss_sum, hi - start):.6f}")

    return mean_loss(loss_sum, end - start)


@torch.no_grad()
def score_stream(model: TGN, events: Events, start: int, end: int, batch_size: int) -> np.ndarray:
    """
    Laundering probability of every transaction in events[start:end], in
    stream order. Each batch is scored before the model sees it, then let
    into the stream state, as it would be in live use.
    """
    model.eval()
    # float64: float32 sigmoids saturate at 1.0 and tie the top scores.
    scores = torch.empty(end - start, dtype=torch.float64, device=model.device)
    progress = Progress("score", end - start)

    for lo in range(start, end, batch_size):
        hi = min(lo + batch_size, end)
        batch = events.rows(slice(lo, hi), model.device)
        logits, msg = model(batch, events)
        scores[lo - start:hi - start] = torch.sigmoid(logits.double())
        model.update_state(batch, msg)
        if progress.due():
            progress.report(hi - start)

    return scores.cpu().numpy()


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


def place_events(events: Events, device: torch.device) -> Events:
    """
    Put the event tensors on the GPU when they fit comfortably. Every batch
    looks up thousands of past transactions; doing that on the CPU costs a
    round trip per batch, and with many CPU threads it was the slowest step.
    """
    if device.type != "cuda":
        return events
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if events.nbytes > 0.5 * free_bytes:
        return events
    return events.to(device)


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
    g.add_argument("--layers", type=int, default=1, choices=(1, 2),
                   help="Attention layers: with 2, an account also sees its neighbours' neighbours")
    g.add_argument("--hop2_size", type=int, default=None,
                   help="With 2 layers: recent transactions looked at for each neighbour "
                        "(default: --neighbor_size)")
    g.add_argument("--no_txn_features", action="store_true",
                   help="Ablation: the classifier does not see the transaction's own attributes")
    g.add_argument("--no_activity_features", action="store_true",
                   help="Ablation: the classifier does not see account activity statistics "
                        "or the local payment structure")
    g.add_argument("--activity_in_attention", action="store_true",
                   help="Also give the attention layers every account's activity statistics")

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--batch_size", type=int, default=1000,
                   help="Transactions per step. The time per step barely depends on it, "
                        "so larger batches are proportionally faster")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--pos_weight", type=float, default=1.0,
                   help="Loss weight of laundering transactions (1 = unweighted)")
    g.add_argument("--max_grad_norm", type=float, default=None,
                   help="Clip the gradient norm at this value (default: no clipping)")
    g.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience (epochs, based on val average precision)")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("run")
    g.add_argument("--device", default=None, help="cuda, cuda:1, cpu ... (default: cuda if available)")
    g.add_argument("--cpu_threads", type=int, default=None,
                   help="CPU threads for PyTorch (default: 2 on a GPU, PyTorch's own choice on a CPU)")
    g.add_argument("--out_dir", default="outputs")
    g.add_argument("--resume", action="store_true",
                   help="Continue from the checkpoint in --out_dir, if there is one")
    g.add_argument("--save_scores", action="store_true",
                   help="Also save the validation and test scores, for plots and "
                        "ensembles without re-running")
    return p.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list | None = None) -> dict:
    args = get_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # On a GPU the CPU only drives the loop. More threads do not help, and they
    # fight each other when several runs share a machine.
    cpu_threads = args.cpu_threads or (2 if device.type == "cuda" else None)
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "tgn_model.pt")
    checkpoint_path = os.path.join(args.out_dir, "tgn_checkpoint.pt")
    metrics_path = os.path.join(args.out_dir, "tgn_metrics.json")
    scores_path = os.path.join(args.out_dir, "tgn_test_scores.npz")
    val_scores_path = os.path.join(args.out_dir, "tgn_val_scores.npz")

    log(f"Device: {device}  (CPU threads: {torch.get_num_threads()})")

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
        num_layers=args.layers,
        use_activity_features=not args.no_activity_features,
        activity_in_attention=args.activity_in_attention,
        hop2_size=args.hop2_size,
    )
    # `stream` is what the model reads; `events` stays on the CPU for the metrics.
    stream = place_events(events, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], dtype=torch.float, device=device),
    )

    log(f"      Parameters   : {sum(p.numel() for p in model.parameters()):,}")
    log(f"      Feature dim  : {model.encoder.out_channels}")
    log(f"      Events on    : {stream.src.device}")

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
            model, stream, 0, train_end, args.batch_size,
            optimizer, criterion, args.max_grad_norm,
        )
        # Val: continue from the end-of-train stream state (don't reset)
        val_prob = score_stream(model, stream, train_end, val_end, args.batch_size)
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
            if args.save_scores:
                # Validation scores of the kept epoch: what a threshold for an
                # ensemble of several runs has to be chosen on.
                np.savez_compressed(val_scores_path + ".tmp.npz", score=val_prob, label=y_val)
                os.replace(val_scores_path + ".tmp.npz", val_scores_path)
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
    replay_neighbors(model, stream, [splits["train"], splits["validation"]], args.batch_size)

    test_prob = score_stream(model, stream, val_end, num_events, args.batch_size)

    # ── 6. Metrics ────────────────────────────────────────────────────────────
    log("[5/5] Computing metrics...")
    threshold = best["threshold"]
    flagged = test_prob >= threshold
    src_test = events.src[val_end:].numpy()
    dst_test = events.dst[val_end:].numpy()
    t_test = events.t[val_end:].numpy()

    test_metrics = classification_metrics(y_test, test_prob, threshold)
    latency = detection_latency(src_test, dst_test, t_test, y_test, flagged)

    # The same again without the quiet days at the end of the file.
    active = ~quiet_day_mask(events.t.numpy())[val_end:]
    active_metrics = classification_metrics(y_test[active], test_prob[active], threshold)
    active_latency = detection_latency(
        src_test[active], dst_test[active], t_test[active], y_test[active], flagged[active],
    )
    quiet = {
        "transactions": int((~active).sum()),
        "laundering": int(y_test[~active].sum()),
    }

    print(format_report("TGN — validation (threshold selected here)", best["validation"]))
    print(format_report("TGN — test", test_metrics, latency))
    if quiet["transactions"]:
        print(format_report(
            f"TGN — test without quiet days ({quiet['transactions']:,} transactions, "
            f"{quiet['laundering']:,} laundering left out)", active_metrics, active_latency,
        ))
    sys.stdout.flush()

    results = {
        "data": {
            "file": args.data,
            "accounts": events.num_nodes,
            "transactions": num_events,
            "laundering": n_launder,
            "splits": split_stats,
            "quiet_days_in_test": quiet,
        },
        "best_epoch": best_epoch,
        "validation": best["validation"],
        "test": test_metrics,
        "detection_latency": latency,
        "test_active_days": active_metrics,
        "detection_latency_active_days": active_latency,
        "history": history,
        "train_seconds": sum(h["seconds"] for h in history),
        "config": {
            **vars(args),
            "device_used": str(device),
            "events_device": str(stream.src.device),
            "cpu_threads_used": torch.get_num_threads(),
        },
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
            src=src_test.astype(np.int32), dst=dst_test.astype(np.int32), t=t_test,
            threshold=threshold, active_day=active,
        )
        log(f"Scores  → {scores_path}")

    return results


if __name__ == "__main__":
    main()
