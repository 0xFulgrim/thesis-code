# Money mule detection on IBM AML transactions

Thesis code: a Temporal Graph Network (TGN) that flags laundering transactions
in the
[IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
dataset, reading the transactions as a stream.

Earlier revisions also contained LightGBM, GraphSAGE and GAT baselines. They
and their results are in the history at commit `7b33ef4`.

## Setup

Python 3.10 or newer. Install PyTorch first, choosing the CUDA or CPU build
for the machine (<https://pytorch.org/get-started/locally/>), then:

```
pip install -r requirements.txt
```

No GPU is required; the script uses CUDA when PyTorch reports it as available.

## Data

Download a `*_Trans.csv` file from the Kaggle page (`HI-Small_Trans.csv` has
about 5 million transactions, the Medium and Large files about 32 and 180
million) and pass it with `--data`. The default is
`data/raw/HI-Small_Trans.csv`.

The first run parses the whole CSV in chunks and caches the result next to it
as `<file>.events.npz`; later runs read the cache. `data/` is git-ignored.

## Running

**1. Check the installation.** The tests run the whole pipeline on a small
synthetic file, on the GPU too if there is one:

```
pip install pytest
pytest -q
```

**2. Check the real data end to end** on its first 200,000 transactions:

```
python src/tgn_model.py --data data/raw/HI-Small_Trans.csv --max_events 200000 --epochs 1
```

**3. Full run.**

```
python -u src/tgn_model.py --data data/raw/HI-Small_Trans.csv --save_scores --resume
```

With `--resume`, rerunning the same command continues an interrupted run from
its last finished epoch; on a finished run it only repeats the evaluation.

Outputs go to `--out_dir` (default `outputs/`):

| File | Contents |
|---|---|
| `tgn_metrics.json` | Validation and test metrics, detection latency, per-epoch history, settings, library versions |
| `tgn_model.pt` | Best epoch: weights and every account's state after the validation stream |
| `tgn_checkpoint.pt` | Last epoch, for `--resume` |
| `tgn_test_scores.npz` | With `--save_scores`: score, label, accounts and time of every test transaction |

Use a separate `--out_dir` for each dataset or setting. Run with `--help` for
all flags. The ones most worth trying:

- `--pos_weight`: loss weight of laundering transactions. The default balances
  the classes completely (on the order of 1000 on these files), which favours
  recall.
- `--batch_size`: larger batches are faster, but transactions in the same
  batch cannot see each other.
- `--no_txn_features`: ablation in which the classifier sees only the two
  account embeddings.
- `--seed`: for repeated runs.

## Method

- Each account has a memory vector, updated by a GRU from its transactions.
  A temporal attention layer embeds an account from its memory and its ten
  most recent counterparties.
- A transaction is scored from the embeddings of its two accounts, its own
  attributes (amounts, currencies, payment format, time of day) and how
  recently each account was last active. Only earlier transactions feed the
  memory and the embeddings.
- Chronological split: first 60% train, next 10% validation, last 30% test.
- The epoch with the best validation average precision is kept.
- The decision threshold maximises F1 on the validation split and is applied
  unchanged to the test split.
- Detection latency is the time from an account's first laundering
  transaction in the test stream to its first alert at or after it. Alerts
  before that are reported separately and do not count as detections.

## What to expect

Measured on a 12-core laptop CPU with synthetic data; nothing here has been
measured on a GPU.

- Parsing: about 600,000 rows per second.
- Training: roughly 6,000 to 17,000 transactions per second, depending on how
  densely the accounts are connected. Scoring is about twice as fast.
- Checkpoints: roughly 0.6 to 0.7 kB per account in each `.pt` file.

## Results

`outputs/tgn_metrics.json` is still from the first version of the model
(commit `7b33ef4`), whose threshold was tuned on the test split. The current
code has not been run on the real dataset yet.

## Repository layout

```
src/
  data.py          CSV parsing, event cache, transaction features
  tgn_model.py     model, training loop, command line
  evaluation.py    threshold selection, metrics, detection latency
tests/             unit tests and end-to-end runs on synthetic data
```
