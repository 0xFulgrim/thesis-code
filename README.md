# Money mule detection on IBM AML transactions

Thesis code comparing four models for detecting money mule accounts in the
[IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
dataset (HI-Small): a tabular baseline, two static graph neural networks, and a
temporal graph network that processes transactions as a stream.

## Models

| Script | Model | Predicts | Train / val / test split |
|---|---|---|---|
| `src/lightgbm_model.py` | LightGBM on per-account aggregate features | Is this account a mule? | Random, stratified, 70 / 10 / 20 |
| `src/graphsage_model.py` | GraphSAGE (2 layers) on the account graph | Is this account a mule? | Random, stratified, 70 / 10 / 20 |
| `src/gat_model.py` | GATv2 (2 layers) on the account graph | Is this account a mule? | Random, stratified, 70 / 10 / 20 |
| `src/tgn_model.py` | Temporal Graph Network | Is this transaction laundering? | Chronological, 60 / 10 / 30 |

An account is labelled a mule if it sends or receives at least one laundering
transaction. In the graph models, accounts are nodes and transactions are
directed edges; GraphSAGE and GAT use the same per-account features as LightGBM
(standardised), so the comparison isolates the effect of adding graph structure.

## Setup

Requires Python 3.10.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux
pip install -r requirements.txt
```

The pinned PyTorch Geometric extension wheels are CPU builds for Windows and
Linux (x86-64), so no GPU is needed. The scripts use CUDA automatically when
PyTorch reports it as available; to train on a GPU, install a CUDA build of
PyTorch and the matching PyG wheels instead.

## Data

Download `HI-Small_Trans.csv` from the Kaggle page linked above and place it at
`data/raw/HI-Small_Trans.csv`, or pass another location with `--data`.
The `data/` directory is git-ignored.

## Running

From the repository root:

```
python src/lightgbm_model.py
python src/graphsage_model.py
python src/gat_model.py
python src/tgn_model.py
```

Each script prints its evaluation and writes `outputs/<model>_metrics.json`
together with the trained model. Hyperparameters are command-line flags; run a
script with `--help` to list them. The random seed is fixed at 42.

## Results

From the metrics files in `outputs/`:

| Model | ROC AUC | Avg. precision | F1 | Precision | Recall |
|---|---|---|---|---|---|
| LightGBM | 0.838 | 0.253 | 0.289 | 0.666 | 0.185 |
| GraphSAGE | 0.908 | 0.370 | 0.405 | 0.624 | 0.300 |
| GATv2 | 0.891 | 0.313 | 0.369 | 0.551 | 0.278 |
| TGN | 0.908 | 0.134 | 0.218 | 0.241 | 0.199 |

- F1, precision, and recall are reported at the threshold that maximises F1,
  which each script selects on its test set.
- The TGN row is per transaction on a chronological split; the other rows are
  per account on a random split. They are not directly comparable.
- Detection latency (TGN only) is the time from an account's first laundering
  transaction in the test stream to the first transaction involving that
  account that the model flags. 631 of the 2,844 mule accounts in the test
  stream were flagged, with a median latency of 0 s and a mean of about 6.5 hours.

`outputs/` also holds the trained GraphSAGE and GAT weights and the LightGBM
booster. The TGN checkpoint is not included; `src/tgn_model.py` writes it to
`outputs/tgn_model.pt`.

## Repository layout

```
src/
  utils.py             shared settings, data loading, features, graph construction, metrics
  static_gnn.py        training pipeline shared by GraphSAGE and GAT
  lightgbm_model.py    tabular baseline
  graphsage_model.py   static GNN baseline
  gat_model.py         static GNN baseline with attention
  tgn_model.py         temporal graph network and detection latency
outputs/               saved metrics and trained models
requirements.txt       pinned dependencies
```
