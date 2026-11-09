# Money mule detection on IBM AML transactions

Thesis code: a Temporal Graph Network (TGN) that flags laundering transactions
in the
[IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
dataset, processing transactions as a stream.

Earlier revisions also contained LightGBM, GraphSAGE and GAT baselines. They
and their results are in the history at commit `7b33ef4`.

## Setup

Requires Python 3.10.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux
pip install -r requirements.txt
```

## Data

Download `HI-Small_Trans.csv` from the Kaggle page linked above and place it at
`data/raw/HI-Small_Trans.csv`, or pass another location with `--data`.
The `data/` directory is git-ignored.

## Running

From the repository root:

```
python src/tgn_model.py
```

The script prints its evaluation and writes `outputs/tgn_metrics.json` and
`outputs/tgn_model.pt`. Hyperparameters are command-line flags; run it with
`--help` to list them.

## Repository layout

```
src/
  utils.py             data loading, temporal graph construction, metrics
  tgn_model.py         temporal graph network and detection latency
outputs/               saved metrics
requirements.txt       pinned dependencies
```
