# Money mule detection on IBM AML transactions

Thesis code: a Temporal Graph Network (TGN) that flags laundering transactions
in the
[IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
dataset, processing transactions as a stream.

Earlier revisions also contained LightGBM, GraphSAGE and GAT baselines. They
and their results are in the history at commit `7b33ef4`.

## Setup

Python 3.10 or newer. Install PyTorch first, choosing the CUDA or CPU build
for the machine (<https://pytorch.org/get-started/locally/>), then:

```
pip install -r requirements.txt
```

## Data

Download a `*_Trans.csv` file (for example `HI-Small_Trans.csv`) from the
Kaggle page and pass it with `--data`; the default is
`data/raw/HI-Small_Trans.csv`. The first run parses the CSV in chunks and
caches the result next to it as `<file>.events.npz`.

## Running

Check the installation first. Both take well under a minute:

```
pip install pytest && pytest -q
python src/tgn_model.py --data data/raw/HI-Small_Trans.csv --max_events 200000 --epochs 1
```

Full run (survives interruption: rerun the same command to continue):

```
python -u src/tgn_model.py --data data/raw/HI-Small_Trans.csv --save_scores --resume
```

Outputs go to `--out_dir` (default `outputs/`): `tgn_metrics.json`,
`tgn_model.pt` (best epoch), `tgn_checkpoint.pt` (for `--resume`) and, with
`--save_scores`, `tgn_test_scores.npz`. Run with `--help` for all flags.
Worth trying first: `--pos_weight` (the default fully balances the classes),
`--batch_size`, and `--no_txn_features` (ablation of the classifier inputs).

## Method

- Chronological split: first 60% train, next 10% validation, last 30% test.
- The epoch with the best validation average precision is kept.
- The decision threshold maximises F1 on the validation split and is then
  applied unchanged to the test split.
- Detection latency: time from an account's first laundering transaction in
  the test stream to its first alert at or after it.

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
