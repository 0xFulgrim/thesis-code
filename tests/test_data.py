import os

import numpy as np
import pandas as pd
import torch
from conftest import CURRENCIES, FORMATS, write_csv

from data import TransactionEncoder, load_events


def quiet(_message):
    pass


def test_load_matches_the_csv(transactions, csv_path):
    events = load_events(csv_path, use_cache=False, log=quiet)
    frame = transactions

    assert events.num_events == len(frame)
    assert int(events.y.sum()) == int(frame["is_laundering"].sum())
    assert events.currencies == sorted(CURRENCIES)
    assert events.formats == sorted(FORMATS)

    # Timestamps: seconds since the epoch, in file order (already sorted).
    expected_t = pd.to_datetime(frame["timestamp"]).to_numpy().astype("datetime64[s]").astype(np.int64)
    assert np.array_equal(events.t.numpy(), expected_t)

    # An account is its bank and account number together.
    from_key = (frame["from_bank"] + "_" + frame["from_account"]).to_numpy()
    to_key = (frame["to_bank"] + "_" + frame["to_account"]).to_numpy()
    assert events.num_nodes == len(set(from_key) | set(to_key))
    key_of_node = {}
    for node, key in zip(
        np.concatenate([events.src.numpy(), events.dst.numpy()]),
        np.concatenate([from_key, to_key]), strict=True,
    ):
        assert key_of_node.setdefault(int(node), key) == key
    assert len(set(key_of_node.values())) == len(key_of_node)

    # Attributes line up with their rows.
    assert np.allclose(events.amount[:, 0].numpy(), np.log1p(frame["amount_paid"]), rtol=1e-6)
    assert np.allclose(events.amount[:, 1].numpy(), np.log1p(frame["amount_received"]), rtol=1e-6)
    paid_in = np.array(events.currencies)[events.cat[:, 0].numpy()]
    received_in = np.array(events.currencies)[events.cat[:, 1].numpy()]
    fmt = np.array(events.formats)[events.cat[:, 2].numpy()]
    assert np.array_equal(paid_in, frame["payment_currency"].to_numpy())
    assert np.array_equal(received_in, frame["receiving_currency"].to_numpy())
    assert np.array_equal(fmt, frame["payment_format"].to_numpy())
    assert np.array_equal(events.y.numpy(), frame["is_laundering"].to_numpy())


def test_bank_ids_compare_by_value_and_accounts_as_text(transactions, tmp_path):
    rows = transactions.iloc[:4].copy()
    rows["from_bank"] = ["010", "10", "0010", "11"]              # three spellings of bank 10
    rows["from_account"] = ["8000A", "8000A", "8000A", "8000A"]
    rows["to_bank"] = ["7", "7", "7", "7"]
    rows["to_account"] = ["0123", "123", "0123", "0123"]         # 0123 and 123 are different accounts
    path = tmp_path / "Banks_Trans.csv"
    write_csv(rows, path)
    events = load_events(str(path), use_cache=False, log=quiet)

    assert events.src.tolist() == [0, 0, 0, 3]
    assert events.dst.tolist() == [1, 2, 1, 1]
    assert events.num_nodes == 4


def test_chunked_parse_equals_single_pass(csv_path):
    whole = load_events(csv_path, use_cache=False, log=quiet)
    chunked = load_events(csv_path, use_cache=False, chunksize=377, log=quiet)
    for name in ("src", "dst", "t", "amount", "cat", "y"):
        assert torch.equal(getattr(whole, name), getattr(chunked, name)), name
    assert whole.num_nodes == chunked.num_nodes


def test_cache_is_reused_and_invalidated(transactions, csv_path):
    messages = []
    first = load_events(csv_path, log=messages.append)
    assert os.path.exists(csv_path + ".events.npz")
    assert not any("Loaded cached" in m for m in messages)

    messages.clear()
    second = load_events(csv_path, log=messages.append)
    assert any("Loaded cached" in m for m in messages)
    for name in ("src", "dst", "t", "amount", "cat", "y"):
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    assert (first.num_nodes, first.currencies, first.formats) == \
           (second.num_nodes, second.currencies, second.formats)

    # A changed CSV must not be served from the old cache.
    write_csv(transactions.iloc[:-10], csv_path)
    messages.clear()
    third = load_events(csv_path, log=messages.append)
    assert not any("Loaded cached" in m for m in messages)
    assert third.num_events == len(transactions) - 10


def test_unsorted_file_is_sorted_keeping_file_order_for_ties(transactions, tmp_path):
    shuffled = transactions.sample(frac=1.0, random_state=1).reset_index(drop=True)
    path = tmp_path / "Shuffled_Trans.csv"
    write_csv(shuffled, path)
    events = load_events(str(path), use_cache=False, log=quiet)

    t = events.t.numpy()
    assert np.all(t[1:] >= t[:-1])
    expected = shuffled.iloc[np.argsort(
        pd.to_datetime(shuffled["timestamp"]).to_numpy(), kind="stable",
    )]
    assert np.allclose(events.amount[:, 0].numpy(), np.log1p(expected["amount_paid"]), rtol=1e-6)


def test_head_shrinks_the_node_count(csv_path):
    events = load_events(csv_path, use_cache=False, log=quiet)
    head = events.head(100)
    assert head.num_events == 100
    assert head.num_nodes == int(max(head.src.max(), head.dst.max())) + 1
    assert head.num_nodes <= events.num_nodes


def test_to_device_keeps_the_stream(csv_path):
    events = load_events(csv_path, use_cache=False, log=quiet)
    moved = events.to(torch.device("cpu"))
    for name in ("src", "dst", "t", "amount", "cat", "y"):
        assert torch.equal(getattr(events, name), getattr(moved, name)), name
    assert (moved.num_nodes, moved.currencies, moved.formats) == \
           (events.num_nodes, events.currencies, events.formats)
    n = events.num_events
    assert events.nbytes == n * (8 + 8 + 8 + 2 * 4 + 3 + 1)


def test_encoder_layout(csv_path):
    events = load_events(csv_path, use_cache=False, log=quiet)
    encoder = TransactionEncoder.fit(events, train_end=events.num_events)
    n_cur, n_fmt = len(events.currencies), len(events.formats)
    assert encoder.out_channels == 2 + 2 * n_cur + n_fmt + 4

    batch = events.rows(slice(0, 500), torch.device("cpu"))
    x = encoder(batch)
    assert x.shape == (500, encoder.out_channels)
    assert torch.isfinite(x).all()

    paid = x[:, 2:2 + n_cur]
    received = x[:, 2 + n_cur:2 + 2 * n_cur]
    fmt = x[:, 2 + 2 * n_cur:2 + 2 * n_cur + n_fmt]
    for block, column in ((paid, 0), (received, 1), (fmt, 2)):
        assert torch.equal(block.sum(dim=1), torch.ones(500))
        assert torch.equal(block.argmax(dim=1), batch.cat[:, column].long())

    flags = x[:, 2 + 2 * n_cur + n_fmt:]
    assert torch.equal(flags[:, 0], (batch.cat[:, 0] != batch.cat[:, 1]).float())
    assert torch.equal(flags[:, 1], (batch.src == batch.dst).float())
    assert torch.allclose(flags[:, 2] ** 2 + flags[:, 3] ** 2, torch.ones(500), atol=1e-5)

    # Amounts are standardised with the statistics of the fitted range.
    full = encoder(events.rows(slice(0, events.num_events), torch.device("cpu")))
    assert torch.allclose(full[:, :2].mean(dim=0), torch.zeros(2), atol=1e-3)
    assert torch.allclose(full[:, :2].std(dim=0), torch.ones(2), atol=1e-3)
