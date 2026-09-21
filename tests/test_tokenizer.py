import random

from specmodel.tokenizer import SPECIALS, Tokenizer, chat_example_ids, chat_prompt_ids, encode_many, learn_merges

TEXTS = [
    "the cat sat on the mat",
    "the dog ran fast and jumped",
    "SELECT count(*) FROM orders WHERE id = 3;",
    'she said "hello world" and laughed',
    "numbers 12345 and 2024-01-05",
    "the the the cat cat",
] * 20


def train():
    return Tokenizer.train(TEXTS, 330, 10**6)


def test_roundtrip_including_unicode_and_specials():
    tok = train()
    text = 'The cat said "hi"! caf\u00e9 \u4e2d\u6587 \U0001f600 SELECT * FROM t WHERE x = 1;'
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert all(i < 256 + len(tok.merges) for i in ids)
    wrapped = "<|user|>hi<|assistant|>"
    plain = tok.encode(wrapped)
    special = tok.encode(wrapped, allow_special=True)
    assert special[0] == tok.special("<|user|>") and special[-1] == tok.special("<|assistant|>")
    assert len(plain) > len(special)
    assert tok.decode(special, skip_special=False) == wrapped
    assert tok.decode(special) == "hi"


def test_vocab_layout_and_save_load(tmp_path):
    tok = train()
    assert tok.vocab_size == 256 + len(tok.merges) + len(SPECIALS)
    assert tok.pad_id < tok.bos_id < tok.eos_id
    tok.save(tmp_path / "tok.json")
    again = Tokenizer.load(tmp_path / "tok.json")
    assert again.merges == tok.merges
    assert again.encode("the cat sat") == tok.encode("the cat sat")


def test_merges_are_deterministic_and_ordered_by_frequency():
    counts = {b"aaab": 5, b"aab": 3, b"cd": 1}
    merges = learn_merges(counts, 3)
    assert merges[0] == (97, 97)
    assert learn_merges(counts, 3) == merges


def test_learned_merges_compress_repeated_words():
    tok = train()
    ids = tok.encode("the cat sat on the mat")
    assert len(ids) < len(b"the cat sat on the mat")


def test_chat_ids_shape_and_truncation():
    tok = train()
    ids = chat_prompt_ids(tok, "system text", "hello there", 64)
    assert ids[0] == tok.bos_id and ids[1] == tok.special("<|system|>") and ids[-1] == tok.special("<|assistant|>")
    assert tok.special("<|user|>") in ids
    long = chat_prompt_ids(tok, "sys", "word " * 500, 64)
    assert len(long) == 64
    ids, labels = chat_example_ids(tok, "sys", "question", "answer", 128)
    assert len(ids) == len(labels)
    prompt_len = labels.index(next(v for v in labels if v != -100))
    assert ids[:prompt_len] == chat_prompt_ids(tok, "sys", "question", 64)
    assert labels[prompt_len:] == ids[prompt_len:]
    assert labels[-1] == tok.eos_id


def test_parallel_encoding_matches_serial(tmp_path):
    tok = train()
    tok.save(tmp_path / "tok.json")
    rng = random.Random(0)
    texts = [" ".join(rng.choice(TEXTS) for _ in range(3)) for _ in range(600)]
    serial = list(encode_many(str(tmp_path / "tok.json"), texts, 1))
    parallel = list(encode_many(str(tmp_path / "tok.json"), texts, 2, batch_size=50))
    assert serial == parallel
    assert serial == [tok.encode(t) for t in texts]
