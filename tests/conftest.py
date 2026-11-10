"""Shared fixtures: a small synthetic file in the IBM AML *_Trans.csv layout."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# The real files repeat the "Account" column name.
HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
)
CURRENCIES = ["US Dollar", "Euro", "Yuan", "Bitcoin", "Rupee"]
FORMATS = ["Cheque", "Credit Card", "ACH", "Wire", "Cash", "Reinvestment", "Bitcoin"]


def synthetic_transactions(n_tx: int = 3000, n_accounts: int = 250, seed: int = 0) -> pd.DataFrame:
    """
    Transactions in time order. About 4% are laundering; those run between a
    small set of mule accounts, mostly by ACH and for larger amounts, so there
    is something to learn.
    """
    rng = np.random.default_rng(seed)
    bank = np.array([f"{b:03d}" for b in rng.integers(1, 40, size=n_accounts)])
    account = np.array([f"8{i:07X}" for i in range(n_accounts)])
    mules = np.flatnonzero(rng.random(n_accounts) < 0.08)

    src = rng.integers(0, n_accounts, size=n_tx)
    dst = rng.integers(0, n_accounts, size=n_tx)
    laundering = rng.random(n_tx) < 0.04
    src[laundering] = rng.choice(mules, size=laundering.sum())
    dst[laundering] = rng.choice(mules, size=laundering.sum())

    paid_in = rng.integers(0, len(CURRENCIES), size=n_tx)
    received_in = np.where(rng.random(n_tx) < 0.9, paid_in, rng.integers(0, len(CURRENCIES), size=n_tx))
    fmt = rng.integers(0, len(FORMATS), size=n_tx)
    fmt[laundering] = np.where(rng.random(laundering.sum()) < 0.8, FORMATS.index("ACH"), fmt[laundering])
    amount = np.round(np.exp(rng.normal(6, 2, size=n_tx)), 2)
    amount[laundering] *= 4

    minutes = np.sort(rng.integers(0, 10 * 24 * 60, size=n_tx))
    timestamp = pd.Timestamp("2022-09-01") + pd.to_timedelta(minutes, unit="m")

    return pd.DataFrame({
        "timestamp": timestamp.strftime("%Y/%m/%d %H:%M"),
        "from_bank": bank[src], "from_account": account[src],
        "to_bank": bank[dst], "to_account": account[dst],
        "amount_received": amount, "receiving_currency": np.array(CURRENCIES)[received_in],
        "amount_paid": amount, "payment_currency": np.array(CURRENCIES)[paid_in],
        "payment_format": np.array(FORMATS)[fmt],
        "is_laundering": laundering.astype(int),
    })


def write_csv(frame: pd.DataFrame, path) -> None:
    with open(path, "w") as f:
        f.write(HEADER)
        frame.to_csv(f, header=False, index=False)


@pytest.fixture(scope="session")
def transactions() -> pd.DataFrame:
    return synthetic_transactions()


@pytest.fixture()
def csv_path(transactions, tmp_path):
    path = tmp_path / "Synthetic_Trans.csv"
    write_csv(transactions, path)
    return str(path)
