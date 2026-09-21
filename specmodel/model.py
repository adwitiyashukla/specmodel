import math

import torch
import torch.nn.functional as F
from torch import nn


class ModelConfig:
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, seq_len, rope_theta=10000.0):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.seq_len = seq_len
        self.rope_theta = rope_theta
        self.head_dim = dim // n_heads
        hidden = int(8 * dim / 3)
        self.hidden_dim = 64 * ((hidden + 63) // 64)

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "seq_len": self.seq_len,
            "rope_theta": self.rope_theta,
        }

    @classmethod
    def from_spec(cls, spec, vocab_size):
        m = spec.model
        return cls(vocab_size, m.dim, m.n_layers, m.n_heads, m.n_kv_heads, m.seq_len, m.rope_theta)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_table(max_len, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class KVCache:
    def __init__(self, config, batch_size, max_len, device, dtype):
        shape = (batch_size, config.n_kv_heads, max_len, config.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layers)]
        self.length = 0
        self.max_len = max_len

    def update(self, layer, k, v):
        t = k.shape[2]
        self.k[layer][:, :, self.length : self.length + t] = k
        self.v[layer][:, :, self.length : self.length + t] = v
        end = self.length + t
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, t):
        self.length += t


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)

    def forward(self, x, cos, sin, mask, cache, layer):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(layer, k, v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None, enable_gqa=True)
        return self.wo(y.transpose(1, 2).reshape(b, t, -1))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = FeedForward(config)

    def forward(self, x, cos, sin, mask, cache, layer):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask, cache, layer)
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_table(config.seq_len, config.head_dim, config.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def rope(self, pos):
        cos = self.rope_cos[pos].unsqueeze(1)
        sin = self.rope_sin[pos].unsqueeze(1)
        return cos, sin

    def forward(self, idx, targets=None, pos=None, mask=None, cache=None, per_token=False):
        t = idx.shape[1]
        if pos is None:
            pos = torch.arange(t, device=idx.device).unsqueeze(0)
        cos, sin = self.rope(pos)
        x = self.tok_emb(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, mask, cache, i)
        logits = self.lm_head(self.norm(x))
        if targets is None:
            return logits
        logits = logits.float()
        if per_token:
            picked = torch.log_softmax(logits, dim=-1).gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            return picked * (targets != -100)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)


def pad_left(seqs, pad_id, device):
    length = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), length), pad_id, dtype=torch.long)
    valid = torch.zeros((len(seqs), length), dtype=torch.bool)
    for i, s in enumerate(seqs):
        idx[i, length - len(s) :] = torch.tensor(s, dtype=torch.long)
        valid[i, length - len(s) :] = True
    return idx.to(device), valid.to(device)


def _filter_logits(logits, top_k, top_p):
    if top_k > 0 and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        drop = (cumulative - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return logits


@torch.no_grad()
def generate_stream(model, prompts, eos_id, pad_id, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
    device = next(model.parameters()).device
    batch = len(prompts)
    idx, valid = pad_left(prompts, pad_id, device)
    if isinstance(max_new_tokens, int):
        max_new_tokens = [max_new_tokens] * batch
    if isinstance(temperature, (int, float)):
        temperature = [float(temperature)] * batch
    limits = torch.tensor(max_new_tokens, device=device)
    temps = torch.tensor(temperature, device=device).unsqueeze(-1)
    prompt_len = idx.shape[1]
    total = min(prompt_len + int(limits.max()), model.config.seq_len)
    cache = KVCache(model.config, batch, total, device, next(model.parameters()).dtype)
    seen = torch.zeros((batch, model.config.vocab_size), dtype=torch.bool, device=device)
    seen.scatter_(1, idx, True)
    seen[:, pad_id] = False
    pos = (valid.cumsum(dim=1) - 1).clamp(min=0)
    eye = torch.eye(prompt_len, dtype=torch.bool, device=device)
    mask = (torch.tril(torch.ones(prompt_len, prompt_len, dtype=torch.bool, device=device)) & valid[:, None, :]) | eye
    logits = model(idx, pos=pos, mask=mask.unsqueeze(1), cache=cache)[:, -1, :]
    cache.advance(prompt_len)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    produced = torch.zeros(batch, dtype=torch.long, device=device)
    next_pos = pos[:, -1:] + 1
    key_valid = valid
    while True:
        logits = torch.log_softmax(logits.float(), dim=-1)
        if repetition_penalty > 1.0:
            logits = torch.where(seen, logits * repetition_penalty, logits)
        greedy = logits.argmax(dim=-1)
        scaled = _filter_logits(logits / temps.clamp(min=1e-5), top_k, top_p)
        sampled = torch.multinomial(torch.softmax(scaled, dim=-1), 1).squeeze(-1)
        nxt = torch.where(temps.squeeze(-1) <= 0.0, greedy, sampled)
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        produced += (~done).long()
        done = done | (nxt == eos_id) | (produced >= limits)
        yield nxt, done
        if bool(done.all()) or cache.length >= total:
            break
        seen.scatter_(1, nxt.unsqueeze(-1), True)
        key_valid = torch.cat([key_valid, torch.ones((batch, 1), dtype=torch.bool, device=device)], dim=1)
        step_mask = key_valid[:, None, None, :]
        logits = model(nxt.unsqueeze(-1), pos=next_pos, mask=step_mask, cache=cache)[:, -1, :]
        cache.advance(1)
        next_pos = next_pos + 1


def generate(model, prompts, eos_id, pad_id, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
    outputs = [[] for _ in prompts]
    finished = [False] * len(prompts)
    for nxt, done in generate_stream(
        model, prompts, eos_id, pad_id, max_new_tokens, temperature, top_k, top_p, repetition_penalty
    ):
        nxt = nxt.tolist()
        for i, token in enumerate(nxt):
            if finished[i]:
                continue
            if token == eos_id:
                finished[i] = True
            else:
                outputs[i].append(token)
        for i, d in enumerate(done.tolist()):
            if d:
                finished[i] = True
    return outputs
