import math

import pytest
import torch

from data import Events, TransactionEncoder, load_events
from tgn_model import (
    TGN,
    AccountActivity,
    StreamMemory,
    TimeEncoder,
    recency,
    replay_neighbors,
    score_stream,
    set_seed,
)


def small_model(
    events: Events,
    use_txn_features: bool = True,
    num_layers: int = 1,
    use_activity_features: bool = True,
) -> TGN:
    set_seed(0)
    return TGN(
        num_nodes=events.num_nodes,
        encoder=TransactionEncoder.fit(events, events.num_events),
        memory_dim=16, time_dim=8, embedding_dim=16, neighbor_size=5,
        use_txn_features=use_txn_features, device=torch.device("cpu"), num_layers=num_layers,
        use_activity_features=use_activity_features,
    )


def tiny_stream(pairs: list) -> Events:
    """A hand-written stream: one transaction per (payer, receiver) pair, a minute apart."""
    n = len(pairs)
    src = torch.tensor([a for a, _ in pairs])
    dst = torch.tensor([b for _, b in pairs])
    return Events(
        src, dst, 1_662_000_000 + 60 * torch.arange(n), torch.rand(n, 2) + 1,
        torch.zeros(n, 4, dtype=torch.uint8), torch.zeros(n, dtype=torch.uint8),
        num_nodes=int(max(src.max(), dst.max())) + 1, currencies=["A"], formats=["X"],
    )


def test_memory_matches_a_per_account_reference():
    """The vectorised store must behave like a plain per-account simulation."""
    generator = torch.Generator().manual_seed(0)
    n_nodes, msg_dim, memory_dim = 12, 5, 8
    memory = StreamMemory(n_nodes, msg_dim, memory_dim, time_dim=6)

    ref_memory = torch.zeros(n_nodes, memory_dim)
    ref_last = [0] * n_nodes
    ref_pending = {}                    # node -> (counterparty, t, msg, paid)

    def read(node):
        if node not in ref_pending:
            return ref_memory[node].clone(), ref_last[node]
        other, t, msg, paid = ref_pending[node]
        gap = t - ref_last[node] if ref_last[node] > 0 else 0     # no previous transaction
        message = torch.cat([
            ref_memory[node], ref_memory[other], msg, torch.tensor([float(paid)]),
            memory.time_enc(torch.tensor([gap]))[0],
        ])
        return memory.gru(message[None], ref_memory[node][None])[0].detach(), t

    now = 1_000
    for _ in range(40):
        b = int(torch.randint(1, 9, (1,), generator=generator))
        src = torch.randint(0, n_nodes, (b,), generator=generator)
        dst = torch.randint(0, n_nodes, (b,), generator=generator)     # self-transfers happen
        t = now + torch.randint(0, 3, (b,), generator=generator).cumsum(0)   # ties happen
        now = int(t[-1])
        msg = torch.randn(b, msg_dim, generator=generator)

        got_memory, got_last = memory(torch.arange(n_nodes))
        for node in range(n_nodes):
            want_memory, want_last = read(node)
            assert torch.allclose(got_memory[node], want_memory, atol=1e-6)
            assert int(got_last[node]) == want_last

        memory.update_state(src, dst, t, msg)

        touched = sorted(set(src.tolist()) | set(dst.tolist()))
        absorbed = {node: read(node) for node in touched}       # all read before any is written
        for node, (value, last) in absorbed.items():
            ref_memory[node], ref_last[node] = value, last
        for j in range(b):               # stream order; the payer's view is written last
            ref_pending[int(dst[j])] = (int(src[j]), int(t[j]), msg[j], False)
            ref_pending[int(src[j])] = (int(dst[j]), int(t[j]), msg[j], True)


def test_activity_matches_a_per_account_reference():
    generator = torch.Generator().manual_seed(0)
    n_nodes = 10
    activity = AccountActivity(n_nodes)
    # role 0 = as payer, 1 = as receiver
    ref = {n: {"count": [0, 0], "amount": [0.0, 0.0], "events": [], "first": 0} for n in range(n_nodes)}

    def expected(node, t):
        r = ref[node]
        recent = [
            sum(0.5 ** ((t - at) / half_life) for at, role in r["events"] if role == wanted)
            for wanted in (0, 1) for half_life in AccountActivity.HALF_LIVES
        ]
        age = t - r["first"] if r["first"] else 0
        return torch.tensor(
            [math.log1p(c) / 5 for c in r["count"]]
            + [math.log1p(x) / 5 for x in recent]
            + [r["amount"][role] / max(r["count"][role], 1) for role in (0, 1)]
            + [math.log1p(age) / 10]
        )

    now = 1_662_000_000
    for _ in range(30):
        b = int(torch.randint(1, 9, (1,), generator=generator))
        src = torch.randint(0, n_nodes, (b,), generator=generator)
        dst = torch.randint(0, n_nodes, (b,), generator=generator)
        t = now + torch.randint(0, 4000, (b,), generator=generator).cumsum(0)
        now = int(t[-1])
        msg = torch.randn(b, 4, generator=generator)

        read_at = now + 500              # the batch above is not counted yet
        got = activity(torch.arange(n_nodes), torch.full((n_nodes,), read_at))
        for node in range(n_nodes):
            assert torch.allclose(got[node], expected(node, read_at), atol=1e-4), node

        activity.update_state(src, dst, t, msg)
        for j in range(b):
            for node, role in ((int(src[j]), 0), (int(dst[j]), 1)):
                r = ref[node]
                r["count"][role] += 1
                r["amount"][role] += float(msg[j, role])
                r["events"].append((int(t[j]), role))
                r["first"] = r["first"] or int(t[j])


def structure_counts(pairs: list, n_past: int) -> list:
    """local_structure of the transactions after the first n_past, as plain counts."""
    events = tiny_stream(pairs)
    model = small_model(events)
    past = events.rows(slice(0, n_past), model.device)
    model.update_state(past, model.encoder(past))
    queries = events.rows(slice(n_past, len(pairs)), model.device)
    return torch.expm1(model.local_structure(queries, events)).round().int().tolist()


def test_local_structure_counts_earlier_payments_each_way():
    counts = structure_counts([(0, 1), (0, 1), (1, 0), (2, 1), (0, 1), (1, 0), (2, 0)], n_past=4)
    # columns 0 and 1: payer paid receiver before / receiver paid payer before
    assert [row[:2] for row in counts] == [[2, 1], [1, 2], [0, 0]]


def test_local_structure_sees_shared_counterparties():
    # 0 pays 9, 9 pays 1 | 3 pays 8, 8 pays 2 | 4 and 5 both pay 7 | 7 pays both 10 and 11
    past = [(0, 9), (9, 1), (3, 8), (8, 2), (4, 7), (5, 7), (7, 10), (7, 11)]
    queries = [
        (0, 1),      # 0 paid 9 and 9 paid 1: a second route from payer to receiver
        (2, 3),      # 3 paid 8 and 8 paid 2: paying 3 back closes the loop
        (4, 5),      # both paid 7
        (10, 11),    # 7 paid both
        (0, 2),      # nothing shared
    ]
    counts = structure_counts(past + queries, n_past=len(past))
    assert [row[2:6] for row in counts] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]


def test_local_structure_counts_distinct_counterparties():
    # Account 0 pays 1 twice, pays 2 and 3 once, and is paid by 4 twice.
    past = [(0, 1), (0, 1), (0, 2), (4, 0), (0, 3), (4, 0)]
    counts = structure_counts(past + [(0, 4)], n_past=len(past))
    # columns 6-9: payer paid / was paid by, receiver paid / was paid by (distinct accounts)
    # small_model keeps the 5 most recent transactions of an account: for 0 that
    # drops its first payment to 1, leaving payments to 1, 2, 3 and from 4.
    assert counts[0][6:] == [3, 1, 1, 0]


def test_attention_is_told_which_way_the_money_went():
    # Same accounts, same times and amounts; only the direction of the first
    # payment differs. The score of the second transaction must notice.
    one_way = tiny_stream([(0, 1), (2, 1)])
    other_way = tiny_stream([(1, 0), (2, 1)])
    other_way.amount.copy_(one_way.amount)
    # The classifier sees the two embeddings only.
    model = small_model(one_way, use_txn_features=False, use_activity_features=False)

    def score_of_second(stream):
        model.reset_state()
        model.eval()
        with torch.no_grad():
            first = stream.rows(slice(0, 1), model.device)
            model.update_state(first, model.encoder(first))
            # Hide the direction from the memory, so only the edge feature can carry it.
            model.memory.pending_paid.zero_()
            return model(stream.rows(slice(1, 2), model.device), stream)[0]

    assert not torch.allclose(score_of_second(one_way), score_of_second(other_way))


def test_time_encoder_starts_with_geometric_frequencies():
    encoder = TimeEncoder(10)
    frequencies = encoder.lin.weight.view(-1)
    assert torch.allclose(frequencies, 10.0 ** -torch.arange(10.0))
    assert torch.allclose(encoder(torch.tensor([0])), torch.ones(1, 10))


@pytest.mark.parametrize("num_layers", [1, 2])
def test_gradients_reach_every_part(csv_path, num_layers):
    events = load_events(csv_path, use_cache=False, log=lambda _m: None)
    model = small_model(events, num_layers=num_layers)
    model.train()
    for lo in (0, 50, 100):                       # let some state build up first
        batch = events.rows(slice(lo, lo + 50), model.device)
        logits, msg = model(batch, events)
        model.update_state(batch, msg)
    logits.sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def test_eval_mode_leaves_the_stream_state_alone(csv_path):
    events = load_events(csv_path, use_cache=False, log=lambda _m: None)
    model = small_model(events)
    score_stream(model, events, 0, 300, 50)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    model.eval()
    model.train()
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_restored_state_continues_the_stream_identically(csv_path):
    events = load_events(csv_path, use_cache=False, log=lambda _m: None)
    model = small_model(events)
    score_stream(model, events, 0, 620, 50)
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    continued = score_stream(model, events, 620, 900, 50)

    model.reset_state()
    model.load_state_dict(snapshot)
    replay_neighbors(model, events, [(0, 620)], 50)
    restored = score_stream(model, events, 620, 900, 50)
    assert (continued == restored).all()


def test_recency():
    t = torch.tensor([1000, 1000, 1000])
    last_update = torch.tensor([0, 1000, 900])
    features = recency(t, last_update)
    assert features[:, 1].tolist() == [1.0, 0.0, 0.0]            # first node was never seen
    assert features[0, 0] == 0.0 and features[1, 0] == 0.0
    assert torch.isclose(features[2, 0], torch.log1p(torch.tensor(100.0)) / 10)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_scores_do_not_depend_on_later_transactions(csv_path, num_layers):
    events = load_events(csv_path, use_cache=False, log=lambda _m: None)
    model = small_model(events, num_layers=num_layers)
    batch_size, cut, end = 50, 600, 900

    def scores(stream: Events):
        model.reset_state()
        return score_stream(model, stream, 0, end, batch_size)

    baseline = scores(events)

    # Rewrite everything from `cut` onwards: other accounts, amounts and labels.
    generator = torch.Generator().manual_seed(1)
    altered = Events(
        events.src.clone(), events.dst.clone(), events.t.clone(), events.amount.clone(),
        events.cat.clone(), events.y.clone(), events.num_nodes, events.currencies, events.formats,
    )
    n_late = events.num_events - cut
    altered.src[cut:] = torch.randint(0, events.num_nodes, (n_late,), generator=generator)
    altered.dst[cut:] = torch.randint(0, events.num_nodes, (n_late,), generator=generator)
    altered.amount[cut:] = torch.rand(n_late, 2, generator=generator) * 10
    altered.y[cut:] = 1 - altered.y[cut:]

    changed = scores(altered)
    assert (baseline[:cut] == changed[:cut]).all()
    assert (baseline[cut:] != changed[cut:]).any()


def test_labels_never_reach_the_model(csv_path):
    events = load_events(csv_path, use_cache=False, log=lambda _m: None)
    model = small_model(events)

    def scores(stream: Events):
        model.reset_state()
        return score_stream(model, stream, 0, 600, 50)

    flipped = Events(
        events.src, events.dst, events.t, events.amount, events.cat,
        1 - events.y, events.num_nodes, events.currencies, events.formats,
    )
    assert (scores(events) == scores(flipped)).all()


def test_ablation_head_ignores_transaction_features(csv_path):
    events = load_events(csv_path, use_cache=False, log=lambda _m: None)
    full = small_model(events, use_txn_features=True)
    ablated = small_model(events, use_txn_features=False)
    extra = full.encoder.out_channels + 4
    assert full.classifier.mlp[0].in_features == ablated.classifier.mlp[0].in_features + extra
