import torch

from data import Events, TransactionEncoder, load_events
from tgn_model import (
    TGN,
    StreamMemory,
    TimeEncoder,
    recency,
    replay_neighbors,
    score_stream,
    set_seed,
)


def small_model(events: Events, use_txn_features: bool = True) -> TGN:
    set_seed(0)
    return TGN(
        num_nodes=events.num_nodes,
        encoder=TransactionEncoder.fit(events, events.num_events),
        memory_dim=16, time_dim=8, embedding_dim=16, neighbor_size=5,
        use_txn_features=use_txn_features, device=torch.device("cpu"),
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


def test_time_encoder_starts_with_geometric_frequencies():
    encoder = TimeEncoder(10)
    frequencies = encoder.lin.weight.view(-1)
    assert torch.allclose(frequencies, 10.0 ** -torch.arange(10.0))
    assert torch.allclose(encoder(torch.tensor([0])), torch.ones(1, 10))


def test_gradients_reach_every_part(csv_path):
    events = load_events(csv_path, use_cache=False, log=lambda _m: None)
    model = small_model(events)
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


def test_scores_do_not_depend_on_later_transactions(csv_path):
    events = load_events(csv_path, use_cache=False, log=lambda _m: None)
    model = small_model(events)
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
