"""
data.py — Turns an IBM AML transaction file into the event stream the TGN reads.

The CSV is parsed in chunks, so the Medium and Large files fit in memory, and
the parsed result is cached next to the CSV so later runs skip the parse.
Only compact per-transaction attributes are stored; TransactionEncoder expands
them into feature vectors one batch at a time.
"""

from __future__ import annotations

import math
import os
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DEFAULT_TRANS_PATH = os.path.join("data", "raw", "HI-Small_Trans.csv")

# Bump when the cached arrays change meaning, so stale caches are re-parsed.
CACHE_VERSION = 1

COLUMNS = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency",
    "amount_paid", "payment_currency",
    "payment_format", "is_laundering",
]
DTYPES = {
    "timestamp": str,
    "from_bank": str, "from_account": str,
    "to_bank": str, "to_account": str,
    "amount_received": np.float64, "receiving_currency": str,
    "amount_paid": np.float64, "payment_currency": str,
    "payment_format": str, "is_laundering": np.int8,
}
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
SECONDS_PER_DAY = 86_400


# ── Event stream ──────────────────────────────────────────────────────────────

class Batch(NamedTuple):
    """Attributes of a set of transactions, all on one device."""
    src: torch.Tensor      # [B]    paying account
    dst: torch.Tensor      # [B]    receiving account
    t: torch.Tensor        # [B]    seconds since the Unix epoch
    amount: torch.Tensor   # [B, 2] log1p(amount paid), log1p(amount received)
    cat: torch.Tensor      # [B, 3] payment currency, receiving currency, payment format
    y: torch.Tensor        # [B]    1 = laundering


@dataclass
class Events:
    """Every transaction of a file in chronological order, kept on the CPU."""
    src: torch.Tensor      # int64
    dst: torch.Tensor      # int64
    t: torch.Tensor        # int64
    amount: torch.Tensor   # float32
    cat: torch.Tensor      # uint8
    y: torch.Tensor        # uint8
    num_nodes: int
    currencies: list[str]
    formats: list[str]

    @property
    def num_events(self) -> int:
        return self.src.numel()

    def rows(self, index: slice | torch.Tensor, device: torch.device) -> Batch:
        """The selected transactions, moved to `device`."""
        if isinstance(index, torch.Tensor):
            index = index.to(self.src.device)
        return Batch(*(
            x[index].to(device)
            for x in (self.src, self.dst, self.t, self.amount, self.cat, self.y)
        ))

    def head(self, n: int) -> Events:
        """The first n transactions, with the node count shrunk to match."""
        n = min(n, self.num_events)
        src, dst = self.src[:n], self.dst[:n]
        num_nodes = int(max(src.max(), dst.max())) + 1 if n else 0
        return Events(src, dst, self.t[:n], self.amount[:n], self.cat[:n],
                      self.y[:n], num_nodes, self.currencies, self.formats)


# ── Feature expansion ─────────────────────────────────────────────────────────

class TransactionEncoder(torch.nn.Module):
    """
    Expands the stored attributes of a transaction into its feature vector:

      standardised log amounts (paid, received)          2
      one-hot payment currency, receiving currency       2 x currencies
      one-hot payment format                             formats
      paid and received in different currencies          1
      account pays itself                                1
      time of day (sin, cos)                             2

    It has no trainable parameters. The amount statistics come from the
    training split only.
    """

    def __init__(
        self,
        num_currencies: int,
        num_formats: int,
        amount_mean: torch.Tensor,
        amount_std: torch.Tensor,
    ):
        super().__init__()
        self.num_currencies = num_currencies
        self.num_formats = num_formats
        self.out_channels = 2 + 2 * num_currencies + num_formats + 4
        self.register_buffer("amount_mean", amount_mean.clone())
        self.register_buffer("amount_std", amount_std.clone())

    @classmethod
    def fit(cls, events: Events, train_end: int) -> TransactionEncoder:
        amount = events.amount[:train_end]
        return cls(
            len(events.currencies), len(events.formats),
            amount.mean(dim=0), amount.std(dim=0).clamp_min(1e-6),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        dtype = batch.amount.dtype
        cat = batch.cat.long()
        paid_in, received_in, fmt = cat[:, 0], cat[:, 1], cat[:, 2]
        time_of_day = (batch.t % SECONDS_PER_DAY).to(dtype) * (2 * math.pi / SECONDS_PER_DAY)
        return torch.cat([
            (batch.amount - self.amount_mean) / self.amount_std,
            F.one_hot(paid_in, self.num_currencies).to(dtype),
            F.one_hot(received_in, self.num_currencies).to(dtype),
            F.one_hot(fmt, self.num_formats).to(dtype),
            (paid_in != received_in).to(dtype).unsqueeze(-1),
            (batch.src == batch.dst).to(dtype).unsqueeze(-1),
            torch.sin(time_of_day).unsqueeze(-1),
            torch.cos(time_of_day).unsqueeze(-1),
        ], dim=-1)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_events(
    path: str = DEFAULT_TRANS_PATH,
    use_cache: bool = True,
    chunksize: int = 1_000_000,
    log: Callable[[str], None] = print,
) -> Events:
    """
    Read a *_Trans.csv file into an Events stream sorted by timestamp.

    An account is identified by its bank and account number together. With
    use_cache, the parsed arrays are stored in `<path>.events.npz` and reused
    for as long as the CSV keeps the same size and modification time.
    """
    cache_path = path + ".events.npz"
    stat = os.stat(path)
    stamp = np.array([CACHE_VERSION, stat.st_size, stat.st_mtime_ns], dtype=np.int64)

    arrays = _read_cache(cache_path, stamp) if use_cache else None
    if arrays is not None:
        log(f"Loaded cached events from {cache_path}")
    else:
        arrays = _parse_csv(path, chunksize, log)
        if use_cache:
            _write_cache(cache_path, stamp, arrays, log)

    return Events(
        src=torch.from_numpy(arrays["src"].astype(np.int64)),
        dst=torch.from_numpy(arrays["dst"].astype(np.int64)),
        t=torch.from_numpy(arrays["t"]),
        amount=torch.from_numpy(arrays["amount"]),
        cat=torch.from_numpy(arrays["cat"]),
        y=torch.from_numpy(arrays["y"]),
        num_nodes=int(arrays["num_nodes"]),
        currencies=arrays["currencies"].tolist(),
        formats=arrays["formats"].tolist(),
    )


def _parse_csv(path: str, chunksize: int, log: Callable[[str], None]) -> dict:
    accounts: dict = {}
    currencies: dict = {}
    formats: dict = {}
    parts: dict = {k: [] for k in ("src", "dst", "t", "amount", "cat", "y")}
    rows = 0

    # na_filter=False: account numbers and names are identifiers, never "missing".
    reader = pd.read_csv(path, header=0, names=COLUMNS, dtype=DTYPES,
                         na_filter=False, chunksize=chunksize)
    for chunk in reader:
        n = len(chunk)
        # Payer and receiver are interleaved row by row, so accounts are
        # numbered in order of first appearance whatever the chunk size.
        node = _encode(_interleave(
            chunk["from_bank"].str.cat(chunk["from_account"], sep="_"),
            chunk["to_bank"].str.cat(chunk["to_account"], sep="_"),
        ), accounts, "account").reshape(n, 2)
        currency = _encode(_interleave(
            chunk["payment_currency"], chunk["receiving_currency"],
        ), currencies, "currency").reshape(n, 2)
        fmt = _encode(chunk["payment_format"].to_numpy(dtype=object), formats, "payment format")
        if max(len(currencies), len(formats)) > 255:
            raise ValueError("too many distinct currencies or payment formats for uint8 codes")

        parts["src"].append(node[:, 0].astype(np.int32))
        parts["dst"].append(node[:, 1].astype(np.int32))
        parts["t"].append(_to_seconds(chunk["timestamp"]))
        parts["amount"].append(np.log1p(np.stack([
            chunk["amount_paid"].to_numpy(), chunk["amount_received"].to_numpy(),
        ], axis=1)).astype(np.float32))
        parts["cat"].append(np.column_stack([currency, fmt]).astype(np.uint8))
        parts["y"].append(chunk["is_laundering"].to_numpy().astype(np.uint8))

        rows += n
        log(f"      parsed {rows:,} rows")

    if rows == 0:
        raise ValueError(f"{path} contains no transactions")

    arrays = {k: np.concatenate(v) for k, v in parts.items()}

    # Codes were handed out in order of first appearance; renumber them
    # alphabetically so the feature layout does not depend on row order.
    currency_names, currency_map = _alphabetical(currencies)
    format_names, format_map = _alphabetical(formats)
    cat = arrays["cat"]
    arrays["cat"] = np.stack([
        currency_map[cat[:, 0]], currency_map[cat[:, 1]], format_map[cat[:, 2]],
    ], axis=1).astype(np.uint8)

    # Stable sort: transactions sharing a timestamp keep their file order.
    t = arrays["t"]
    if np.any(t[1:] < t[:-1]):
        order = np.argsort(t, kind="stable")
        arrays = {k: v[order] for k, v in arrays.items()}

    arrays["num_nodes"] = np.int64(len(accounts))
    arrays["currencies"] = np.array(currency_names)
    arrays["formats"] = np.array(format_names)
    return arrays


def _interleave(first: pd.Series, second: pd.Series) -> np.ndarray:
    """first[0], second[0], first[1], second[1], ..."""
    return np.stack([first.to_numpy(dtype=object), second.to_numpy(dtype=object)], axis=1).ravel()


def _encode(values: np.ndarray, vocabulary: dict, what: str) -> np.ndarray:
    """Map values to integer codes, extending `vocabulary` with unseen values."""
    codes, uniques = pd.factorize(values)
    if (codes < 0).any():
        raise ValueError(f"missing {what} value")
    lookup = np.fromiter(
        (vocabulary.setdefault(value, len(vocabulary)) for value in uniques),
        dtype=np.int64, count=len(uniques),
    )
    return lookup[codes]


def _alphabetical(vocabulary: dict) -> tuple[list, np.ndarray]:
    """Names in alphabetical order, and the map from old codes to new ones."""
    names = sorted(vocabulary)
    remap = np.empty(len(names), dtype=np.int64)
    for new_code, name in enumerate(names):
        remap[vocabulary[name]] = new_code
    return names, remap


def _to_seconds(timestamps: pd.Series) -> np.ndarray:
    try:
        parsed = pd.to_datetime(timestamps, format=TIMESTAMP_FORMAT)
    except ValueError:
        parsed = pd.to_datetime(timestamps, format="mixed")
    # Convert through an explicit unit: the resolution pandas picks for parsed
    # datetimes differs between versions.
    return parsed.to_numpy().astype("datetime64[s]").astype(np.int64)


# ── Cache ─────────────────────────────────────────────────────────────────────

def _read_cache(cache_path: str, stamp: np.ndarray) -> dict | None:
    if not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path) as cached:
            if not np.array_equal(cached["stamp"], stamp):
                return None
            return {name: cached[name] for name in cached.files}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None


def _write_cache(
    cache_path: str, stamp: np.ndarray, arrays: dict, log: Callable[[str], None],
) -> None:
    tmp_path = cache_path + ".tmp.npz"
    try:
        np.savez(tmp_path, stamp=stamp, **arrays)
        os.replace(tmp_path, cache_path)
        log(f"Cached parsed events at {cache_path}")
    except OSError as error:
        # A read-only data directory is fine; the next run just parses again.
        log(f"Could not write the event cache ({error}); continuing without it")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
