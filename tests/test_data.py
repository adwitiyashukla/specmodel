import json

import numpy as np
from synthetic import sql_rows, write_sql_parquet

from specmodel.data import build_records, iter_rows, minhash_duplicates, train_tokenizer, write_shards
from specmodel.spec import load_spec
from specmodel.tasks import get_task
from specmodel.tokenizer import Tokenizer


def test_minhash_finds_near_duplicates_only():
    base = " ".join(f"word{i} thing{i % 7} item{i % 5}" for i in range(40))
    near = base + " and then goes home"
    other = " ".join(f"other{i} column{i % 3} table{i % 4}" for i in range(40))
    texts = [base, other, near, base]
    dups, threshold = minhash_duplicates(texts, 64, 8, 5, 1, 1)
    assert 0.7 < threshold < 0.85
    assert dups == {2, 3}
    parallel, _ = minhash_duplicates(texts * 300, 64, 8, 5, 1, 2)
    assert 3 in parallel and 0 not in parallel and 1 not in parallel


def test_build_records_dedups_and_splits(story_spec, tmp_path):
    spec = load_spec(story_spec)
    stats = build_records(spec, get_task("story"), tmp_path, tmp_path / "cache")
    assert stats["sources"]["tinystories"]["rows"] == 440
    assert stats["exact_duplicates"] >= 20
    assert stats["near_duplicates"] >= 1
    assert stats["train"] > 0 and stats["val"] > 0
    assert (
        stats["train"] + stats["val"] + stats["exact_duplicates"] + stats["near_duplicates"] + stats["filtered"] == 440
    )
    with open(tmp_path / "data" / "train.jsonl", encoding="utf-8") as f:
        rec = json.loads(f.readline())
    assert set(rec) == {"text", "prompt", "response", "meta"}
    tok = train_tokenizer(spec, tmp_path)
    assert tok.vocab_size <= spec.tokenizer.vocab_size
    meta = write_shards(spec, tmp_path)
    assert meta["train_docs"] == stats["train"] and meta["val_docs"] == stats["val"]
    tokens = np.fromfile(tmp_path / "shards" / "train_000.bin", dtype=np.uint16)
    assert len(tokens) == meta["train_tokens"]
    assert int((tokens == tok.eos_id).sum()) == stats["train"]
    again = Tokenizer.load(tmp_path / "tokenizer" / "tokenizer.json")
    first_doc = tokens[: int(np.argmax(tokens == tok.eos_id))]
    assert again.decode(first_doc) == rec["text"]


def test_parquet_rows_and_fixed_splits(sql_spec, tmp_path):
    spec = load_spec(sql_spec)
    rows = list(iter_rows("parquet", spec.data.sources[1]["path"]))
    assert len(rows) == 40 and "sql_context" in rows[0]
    stats = build_records(spec, get_task("sql"), tmp_path, tmp_path / "cache")
    assert stats["val"] == 40 - stats["exact_duplicates"] or stats["val"] <= 40
    assert stats["train"] > 200


def test_jsonl_and_text_formats(tmp_path):
    rows = sql_rows(3)
    path = tmp_path / "rows.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    assert [r["id"] for r in iter_rows("jsonl", path)] == [0, 1, 2]
    text = tmp_path / "docs.txt"
    text.write_text("first story<|endoftext|>second story<|endoftext|>", encoding="utf-8")
    assert [r["story"] for r in iter_rows("text", text)] == ["first story", "second story"]
    write_sql_parquet(tmp_path / "p.parquet", 5)
    assert len(list(iter_rows("parquet", tmp_path / "p.parquet"))) == 5
