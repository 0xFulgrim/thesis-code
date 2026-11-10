# Money mule detection on IBM AML transactions

Thesis code: a Temporal Graph Network (TGN) that flags laundering transactions
in the
[IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
dataset. It reads the transactions as a stream: every transaction is scored
with only what happened before it.

Earlier revisions also contained LightGBM, GraphSAGE and GAT baselines. They
and their results are in the history at commit `7b33ef4`.

## Results

All numbers are for the test split (the last 30% of the stream), with the
decision threshold chosen on the validation split. AP is average precision.

**HI-Small** (test split: 1,523,504 transactions, 2,321 of them laundering)

| Model | Test AP | F1 | Precision | Recall | AP without quiet days | F1 without quiet days |
|---|---|---|---|---|---|---|
| First version of this code (threshold tuned on the test split) | 0.134 | 0.218 | 0.241 | 0.199 | | |
| Current model, one run (mean of 5 seeds) | 0.661 | 0.662 | 0.732 | 0.605 | 0.546 | 0.564 |
| Current model, average of the 5 runs | 0.688 | 0.678 | 0.709 | 0.650 | 0.573 | 0.576 |

Between seeds, AP varies by about 0.006 and F1 by about 0.013 (standard
deviation). F1 depends on the validation-chosen threshold more than AP does:
an average of four earlier runs scored AP 0.682 and F1 0.703.

The averaged model flags 2,129 transactions, 1,509 of them correctly, and
raises an alert on 1,606 of the 2,844 accounts involved in laundering. For at
least nine in ten of those accounts the alert comes in the same minute as
their first laundering transaction in the test stream. See "Reading the
scores" for what the quiet days are and why recall cannot get much past 0.7
on this file.

**HI-Medium** (test split: 9,569,472 transactions, 14,147 of them
laundering), one run with the same settings at batch 4000:

| Model | Test AP | F1 | Precision | Recall | AP without quiet days | F1 without quiet days |
|---|---|---|---|---|---|---|
| Current model, one run | 0.726 | 0.745 | 0.886 | 0.642 | 0.623 | 0.661 |

It flags 10,250 transactions, 9,082 of them correctly, and raises an alert on
9,787 of the 17,121 accounts involved in laundering. Training took 30 epochs
and 153 minutes on an H100 that other runs were sharing.

**What each change contributed** on HI-Small, one run each with the same seed:

| | Validation AP | Test AP | Test F1 |
|---|---|---|---|
| Balanced class weight and gradient clipping (the first version's training settings) | 0.332 | 0.371 | 0.388 |
| Unweighted loss, no clipping | 0.528 | 0.576 | 0.599 |
| + payment direction and same-bank flag | 0.530 | 0.561 | 0.580 |
| + account activity statistics and pair history | 0.562 | 0.602 | 0.619 |
| + shared-counterparty and fan-in/fan-out counts | 0.562 | 0.612 | 0.632 |
| 20 neighbours instead of 10 | 0.578 | 0.641 | 0.656 |
| Two attention layers, 30 neighbours | 0.607 | 0.655 | 0.669 |
| Two attention layers, 50 neighbours (the default) | 0.624 | 0.667 | 0.682 |

The metrics files behind the first two tables are in `outputs/`: `hi-small/`
for the five runs and their average, `hi-medium/`, and `first-version/` for
the first version.

Differences of about 0.01 in validation AP and 0.02 in test F1 are within the
spread between seeds; in particular 30 and 50 neighbours are not
distinguishable (test AP 0.655 over four seeds against 0.661 over five).
Leaving the transaction's own attributes out of the classifier costs about
0.08 validation AP.

## Setup

Python 3.10 or newer. Install PyTorch first, choosing the CUDA or CPU build
for the machine (<https://pytorch.org/get-started/locally/>), then:

```
pip install -r requirements.txt
```

No GPU is required; the script uses CUDA when PyTorch reports it as available.

## Data

Download a `*_Trans.csv` file from the Kaggle page and pass it with `--data`.
The default is `data/raw/HI-Small_Trans.csv`.

| File | Transactions | Laundering | Accounts |
|---|---|---|---|
| `HI-Small_Trans.csv` | 5,078,345 | 5,177 | 515,088 |
| `HI-Medium_Trans.csv` | 31,898,238 | 35,230 | 2,077,023 |
| `HI-Large_Trans.csv` | 179,702,229 | 225,546 | 2,116,168 |

The first run parses the whole CSV in chunks (about 600,000 rows per second)
and caches the result next to it as `<file>.events.npz`; later runs read the
cache. `data/` is git-ignored.

## Running

**1. Check the installation.** The tests run the whole pipeline on a small
synthetic file, on the GPU too if there is one:

```
pip install pytest
pytest -q
```

**2. Check the real data end to end** on its first million transactions.
(Laundering is rare at the start of the files: the first 200,000 transactions
of HI-Small contain only 10 cases, the first million 278.)

```
python src/tgn_model.py --data data/raw/HI-Small_Trans.csv --max_events 1000000 --epochs 1
```

**3. Full run.**

```
python -u src/tgn_model.py --data data/raw/HI-Small_Trans.csv --out_dir outputs/small --save_scores --resume
python -u src/tgn_model.py --data data/raw/HI-Large_Trans.csv --out_dir outputs/large --save_scores --resume --batch_size 4000
```

With `--resume`, rerunning the same command continues an interrupted run from
its last finished epoch; on a finished run it only repeats the evaluation.
Use a separate `--out_dir` for each dataset, seed or setting.

Outputs in `--out_dir`:

| File | Contents |
|---|---|
| `tgn_metrics.json` | Validation and test metrics, detection latency, per-epoch history, settings, library versions |
| `tgn_model.pt` | Best epoch: weights and every account's state after the validation stream |
| `tgn_checkpoint.pt` | Last epoch, for `--resume` |
| `tgn_val_scores.npz`, `tgn_test_scores.npz` | With `--save_scores`: the score and label of every validation and test transaction |

**4. Look at the results.**

```
python src/summarize_runs.py outputs                    # one line per finished run, best validation first
python src/ensemble.py outputs/seed1 outputs/seed2 ...  # evaluate the average of several runs
python src/scheme_report.py --data data/raw/HI-Small_Trans.csv \
    --patterns data/raw/HI-Small_Patterns.txt --run outputs/small   # results by laundering scheme
```

### Settings

Run with `--help` for all flags. The defaults are the settings that did best
on HI-Small validation. The ones that matter:

| Flag | Default | Notes |
|---|---|---|
| `--batch_size` | 1000 | The time per step barely depends on it, so larger batches are proportionally faster, but transactions in one batch cannot see each other. Use 4000 on the Medium and Large files, where a batch of 4000 covers about three minutes of the stream. |
| `--lr` | 3e-4 | |
| `--layers` | 2 | Attention layers. With 2, an account also sees its neighbours' neighbours. |
| `--neighbor_size` | 50 | Most recent transactions kept and looked at for each account. |
| `--pos_weight` | 1 | Loss weight of laundering transactions. Larger weights did worse. |
| `--max_grad_norm` | off | Gradient clipping. See the warning below. |
| `--seed` | 42 | |
| `--no_txn_features`, `--no_activity_features` | | Ablations of the classifier inputs. |

Do not combine a large `--pos_weight` with gradient clipping. With a balanced
weight (about 1,300 here) and clipping at 1.0, which the first version of this
code used, the gradients of the few batches that contain laundering were scaled
down by a factor of 100 to 10,000, and the model stalled.

### Several runs on one GPU

One run uses only part of a large GPU. To run several side by side, start
NVIDIA's multi-process service first (`nvidia-cuda-mps-control -d`); without
it they take turns and each slows down by about as much as there are runs.
The script limits itself to two CPU threads on a GPU for the same reason.
Watch the GPU memory: a run that cannot allocate fails.

## Method

- **Memory.** Each account has a memory vector, updated by a GRU from the
  transactions it takes part in (Rossi et al. 2020). The update knows whether
  the account paid or received.
- **Attention.** Two temporal attention layers embed an account from its
  memory, its 50 most recent transactions and those of its counterparties.
  Each past transaction is marked with the direction the money went.
- **Classifier.** A transaction is scored from the embeddings of its two
  accounts together with
  - its own attributes: amounts, currencies, payment format, time of day,
    whether both accounts are at the same bank;
  - how long ago each account was last active;
  - running activity statistics of each account: payments made and received
    in total and over the last hour, day and week, mean amounts, age;
  - the payment pattern around it: earlier payments between the two accounts
    in each direction, third accounts both dealt with (which separates a
    second route for the money, a closed loop, fan-in and fan-out), and how
    many different accounts each recently paid and was paid by.
- **No look-ahead.** A batch is scored first and only then let into the
  memory, the statistics and the neighbour lists. Tests check that scores do
  not change when later transactions or any labels are changed.
- **Split.** Chronological: first 60% train, next 10% validation, last 30% test.
- **Selection.** The epoch with the best validation average precision is
  kept. The decision threshold maximises F1 on the validation split and is
  applied unchanged to the test split.
- **Detection latency.** The time from an account's first laundering
  transaction in the test stream to its first alert at or after it.

## Reading the scores

Two properties of the data set limits on what any model can score here.

**Quiet days.** Each file ends with a run of nearly empty days on which most
remaining transactions are laundering: in HI-Small, the last 8 days hold 1,108
transactions, 655 of them laundering. They all fall in the test split and
flatter every model, so the test metrics are reported with and without them.

**Laundering outside the listed schemes.** The dataset's `*_Patterns.txt` file
lists the laundering transactions that belong to a scheme (fan-out, cycle,
scatter-gather and so on). In HI-Small that is 3,209 of the 5,177 laundering
transactions. The other 1,968 belong to no listed scheme, involve almost
entirely different accounts, and look like ordinary payments; the model
catches about 1% of them. They are 28% of the laundering in the test split,
which caps recall at about 0.72.

Of the laundering that does belong to a scheme, one run of the model catches
84% in the test split (`src/scheme_report.py`):

| Scheme | Laundering transactions | Caught |
|---|---|---|
| Gather-scatter | 449 | 93% |
| Scatter-gather | 328 | 92% |
| Fan-in | 166 | 88% |
| Fan-out | 170 | 86% |
| Random | 104 | 75% |
| Cycle | 147 | 73% |
| Stack | 219 | 70% |
| Bipartite | 97 | 60% |
| No listed scheme | 641 | 1% |

It catches 72% of the transactions that open a scheme, 88% of those from the
sixth onwards, and at least one transaction in 219 of the 226 schemes that
reach into the test split. Every laundering transaction it catches was paid
by ACH, the format of nearly all scheme laundering.

## Speed

Measured on one NVIDIA H100, alone on the GPU, in the middle of the HI-Small
stream, with two layers and 30 neighbours:

| | Time per step | Transactions per second |
|---|---|---|
| Batch 1000, before optimizing the step | 18.9 ms | 53,000 |
| Last attention layer only on the edges it needs | 17.7 ms | 56,000 |
| No data-dependent shapes, less duplicate work | 16.8 ms | 59,000 |
| Batch 4000 | 23.8 ms | 168,000 |

With 50 neighbours a step at batch 4000 takes 25.1 ms, and 22.3 ms with TF32
matrix products (180,000 transactions per second), which the script enables.
Half precision (bfloat16) was slower.

A step launches about 600 small GPU operations whatever the batch size, so
the batch size is what sets the speed. Scoring the validation and test
streams is faster than training, since there is no backward pass.

## Repository layout

```
src/
  data.py            CSV parsing, event cache, transaction features
  tgn_model.py       model, training loop, command line
  evaluation.py      threshold selection, metrics, detection latency, quiet days
  summarize_runs.py  table of finished runs
  ensemble.py        evaluate the average of several runs
  scheme_report.py   results by laundering scheme
tests/               unit tests and end-to-end runs on synthetic data
outputs/             metrics files of the reported runs
```
