import torch

from specmodel.export import quantize_int8
from specmodel.model import KVCache, ModelConfig, Transformer, generate, pad_left

PAD, EOS = 352, 354


def model(seed=0):
    torch.manual_seed(seed)
    return Transformer(ModelConfig(vocab_size=360, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, seq_len=48)).eval()


def test_shapes_and_losses():
    m = model()
    x = torch.randint(0, 300, (3, 16))
    assert m(x).shape == (3, 16, 360)
    loss = m(x, targets=x)
    assert loss.dim() == 0 and loss > 0
    per_token = m(x, targets=x, per_token=True)
    assert per_token.shape == (3, 16)
    masked = x.clone()
    masked[:, :8] = -100
    assert torch.all(m(x, targets=masked, per_token=True)[:, :8] == 0)
    assert m.num_params() == sum(p.numel() for p in m.parameters())
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()


def test_cached_padded_attention_matches_plain_forward():
    m = model()
    prompts = [[5, 6, 7, 8, 9, 10], [11, 12, 13], [20, 21, 22, 23]]
    with torch.no_grad():
        idx, valid = pad_left(prompts, PAD, "cpu")
        b, t = idx.shape
        pos = (valid.cumsum(1) - 1).clamp(min=0)
        mask = (torch.tril(torch.ones(t, t, dtype=torch.bool)) & valid[:, None, :]) | torch.eye(t, dtype=torch.bool)
        cache = KVCache(m.config, b, t + 2, "cpu", torch.float32)
        out = m(idx, pos=pos, mask=mask.unsqueeze(1), cache=cache)
        cache.advance(t)
        key_valid = torch.cat([valid, torch.ones(b, 1, dtype=torch.bool)], 1)
        step = m(torch.full((b, 1), 42), pos=pos[:, -1:] + 1, mask=key_valid[:, None, None, :], cache=cache)
        for i, p in enumerate(prompts):
            assert torch.allclose(out[i, -1], m(torch.tensor([p]))[0, -1], atol=1e-4)
            assert torch.allclose(step[i, -1], m(torch.tensor([p + [42]]))[0, -1], atol=1e-4)


def test_generate_matches_naive_greedy_and_respects_limits():
    m = model(3)
    prompts = [[5, 6, 7, 8, 9, 10], [11, 12, 13], [20, 21, 22, 23]]
    outs = generate(m, prompts, EOS, PAD, [6, 3, 5], 0.0, 0, 1.0, 1.0)
    assert [len(o) for o in outs] == [6, 3, 5]
    for p, o in zip(prompts, outs):
        ids = list(p)
        for _ in range(len(o)):
            ids.append(int(m(torch.tensor([ids]))[0, -1].argmax()))
        assert o == ids[len(p) :]


class Boost(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)
        out[..., EOS] += 100.0
        return out


def test_generate_stops_at_eos_and_clips_to_context():
    m = model(4)
    m.lm_head = Boost(m.lm_head)
    outs = generate(m, [[1, 2, 3], [4, 5]], EOS, PAD, 20, 0.0, 0, 1.0, 1.0)
    assert outs == [[], []]
    m = model(5)
    outs = generate(m, [list(range(1, 46)), [3, 4]], EOS, PAD, 30, 0.7, 10, 0.9, 1.2)
    assert all(len(o) <= 4 for o in outs)


def test_sampling_is_seeded_and_penalty_changes_output():
    m = model(6)
    torch.manual_seed(1)
    a = generate(m, [[1, 2, 3]], EOS, PAD, 12, 1.0, 20, 0.95, 1.0)
    torch.manual_seed(1)
    b = generate(m, [[1, 2, 3]], EOS, PAD, 12, 1.0, 20, 0.95, 1.0)
    assert a == b
    greedy = generate(m, [[1, 2, 3]], EOS, PAD, 12, 0.0, 0, 1.0, 1.0)[0]
    penalized = generate(m, [[1, 2, 3]], EOS, PAD, 12, 0.0, 0, 1.0, 2.0)[0]
    assert len(set(greedy)) < 12
    assert penalized != greedy


def test_int8_quantization_error_is_small():
    w = torch.randn(32, 64) * 0.1
    q, scale = quantize_int8(w)
    assert q.dtype == torch.int8 and scale.shape == (32,)
    back = q.float() * scale[:, None]
    assert (back - w).abs().max() < w.abs().max() / 100
