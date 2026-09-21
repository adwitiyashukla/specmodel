import heapq
import json
import re
from collections import Counter, defaultdict
from itertools import pairwise
from multiprocessing import Pool

PATTERN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>"]
NO_RANK = 1 << 60


def _merge(ids, pair, new_id):
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def learn_merges(word_counts, n_merges):
    words = [list(w) for w in word_counts]
    freqs = list(word_counts.values())
    pair_counts = Counter()
    where = defaultdict(set)
    for i, w in enumerate(words):
        for p in pairwise(w):
            pair_counts[p] += freqs[i]
            where[p].add(i)
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    while len(merges) < n_merges and heap:
        neg, pair = heapq.heappop(heap)
        if pair_counts.get(pair, 0) != -neg or neg == 0:
            continue
        new_id = 256 + len(merges)
        merges.append(pair)
        touched = set()
        for i in list(where[pair]):
            w = words[i]
            f = freqs[i]
            for q in pairwise(w):
                pair_counts[q] -= f
                where[q].discard(i)
                touched.add(q)
            w = _merge(w, pair, new_id)
            words[i] = w
            for q in pairwise(w):
                pair_counts[q] += f
                where[q].add(i)
                touched.add(q)
        for q in touched:
            if q != pair and pair_counts[q] > 0:
                heapq.heappush(heap, (-pair_counts[q], q))
        del pair_counts[pair]
        del where[pair]
    return merges


class Tokenizer:
    def __init__(self, merges, specials=None):
        self.merges = [tuple(m) for m in merges]
        self.specials = list(specials or SPECIALS)
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        base = 256 + len(self.merges)
        self.special_ids = {s: base + i for i, s in enumerate(self.specials)}
        self.special_pattern = re.compile("(" + "|".join(re.escape(s) for s in self.specials) + ")")
        self.cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.specials)

    @property
    def pad_id(self):
        return self.special_ids["<|pad|>"]

    @property
    def bos_id(self):
        return self.special_ids["<|bos|>"]

    @property
    def eos_id(self):
        return self.special_ids["<|eos|>"]

    def special(self, name):
        return self.special_ids[name]

    def _encode_chunk(self, chunk):
        cached = self.cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            best = min(pairwise(ids), key=lambda p: self.ranks.get(p, NO_RANK))
            rank = self.ranks.get(best)
            if rank is None:
                break
            ids = _merge(ids, best, 256 + rank)
        if len(self.cache) < 1_000_000:
            self.cache[chunk] = ids
        return ids

    def encode_plain(self, text):
        out = []
        for chunk in PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text, allow_special=False):
        if not allow_special:
            return self.encode_plain(text)
        out = []
        for part in self.special_pattern.split(text):
            if not part:
                continue
            if part in self.special_ids:
                out.append(self.special_ids[part])
            else:
                out.extend(self.encode_plain(part))
        return out

    def decode(self, ids, skip_special=True):
        parts = []
        specials = {v: k for k, v in self.special_ids.items()}
        for i in ids:
            i = int(i)
            if i in specials:
                if not skip_special:
                    parts.append(specials[i].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges, "specials": self.specials}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["merges"], data["specials"])

    @classmethod
    def train(cls, texts, vocab_size, sample_bytes):
        counts = Counter()
        seen = 0
        for text in texts:
            counts.update(chunk.encode("utf-8") for chunk in PATTERN.findall(text))
            seen += len(text)
            if seen >= sample_bytes:
                break
        n_merges = vocab_size - 256 - len(SPECIALS)
        merges = learn_merges(counts, n_merges)
        return cls(merges, SPECIALS)


def chat_prompt_ids(tok, system, user, max_len):
    ids = [tok.bos_id]
    if system:
        ids.append(tok.special("<|system|>"))
        ids.extend(tok.encode_plain(system))
    ids.append(tok.special("<|user|>"))
    tail = [tok.special("<|assistant|>")]
    user_ids = tok.encode_plain(user)
    budget = max(max_len - len(ids) - len(tail), 8)
    if len(user_ids) > budget:
        keep = budget // 2
        user_ids = user_ids[:keep] + user_ids[-(budget - keep) :]
    return ids + user_ids + tail


def chat_example_ids(tok, system, user, response, max_len):
    prompt = chat_prompt_ids(tok, system, user, max_len - 64)
    answer = tok.encode_plain(response) + [tok.eos_id]
    ids = (prompt + answer)[:max_len]
    labels = ([-100] * len(prompt) + answer)[:max_len]
    return ids, labels


_worker_tokenizer = None


def _init_worker(path):
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer.load(path)


def _encode_batch(texts):
    return [_worker_tokenizer.encode_plain(t) for t in texts]


def encode_many(tokenizer_path, texts, workers, batch_size=256):
    if workers <= 1:
        tok = Tokenizer.load(tokenizer_path)
        for text in texts:
            yield tok.encode_plain(text)
        return

    def batches():
        batch = []
        for text in texts:
            batch.append(text)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    with Pool(workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
        for encoded in pool.imap(_encode_batch, batches(), chunksize=4):
            yield from encoded
