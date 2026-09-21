import hashlib
import io
import json
import os
import re
import tarfile
import zlib
from contextlib import ExitStack
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import requests

from specmodel.tokenizer import Tokenizer, encode_many

MERSENNE = (1 << 61) - 1
SHARD_TOKENS = 50_000_000


def download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done % (100 << 20) < (1 << 20):
                    print(f"download {dest.name} {done / total:.0%}", flush=True)
    os.replace(tmp, dest)
    return dest


def source_path(source, cache_dir):
    if source.get("path"):
        return Path(source["path"])
    name = source["url"].rsplit("/", 1)[-1]
    return download(source["url"], Path(cache_dir) / source["name"] / name)


def iter_rows(fmt, path):
    if fmt == "tinystories":
        with tarfile.open(path, "r:*") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                yield from json.load(io.TextIOWrapper(tar.extractfile(member), encoding="utf-8"))
    elif fmt == "parquet":
        import pyarrow.parquet as pq

        table = pq.ParquetFile(path)
        for batch in table.iter_batches(batch_size=4096):
            yield from batch.to_pylist()
    elif fmt == "jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif fmt == "text":
        with open(path, encoding="utf-8") as f:
            for doc in f.read().split("<|endoftext|>"):
                if doc.strip():
                    yield {"story": doc.strip(), "text": doc.strip()}
    else:
        raise ValueError(fmt)


def _fingerprint(text):
    return hashlib.sha1(re.sub(r"\s+", " ", text.lower()).strip().encode("utf-8")).hexdigest()


def _perms(num_perm, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 1 << 32, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64)
    return a, b


def _signature(text, shingle, a, b):
    words = text.lower().split()
    if len(words) <= shingle:
        grams = {" ".join(words)}
    else:
        grams = {" ".join(words[i : i + shingle]) for i in range(len(words) - shingle + 1)}
    hashes = np.fromiter((zlib.crc32(g.encode("utf-8")) for g in grams), dtype=np.uint64, count=len(grams))
    values = (a[:, None] * hashes[None, :] + b[:, None]) % MERSENNE
    return values.min(axis=1).astype(np.uint32)


_sig_args = None


def _init_sig(shingle, a, b):
    global _sig_args
    _sig_args = (shingle, a, b)


def _sig_batch(texts):
    shingle, a, b = _sig_args
    return np.stack([_signature(t, shingle, a, b) for t in texts])


def minhash_duplicates(texts, num_perm, bands, shingle, seed, workers):
    a, b = _perms(num_perm, seed)
    rows = num_perm // bands
    threshold = (1.0 / bands) ** (1.0 / rows)
    chunks = [texts[i : i + 1000] for i in range(0, len(texts), 1000)]
    if workers > 1 and len(chunks) > 1:
        with Pool(workers, initializer=_init_sig, initargs=(shingle, a, b)) as pool:
            parts = pool.map(_sig_batch, chunks)
    else:
        _init_sig(shingle, a, b)
        parts = [_sig_batch(c) for c in chunks]
    sigs = np.concatenate(parts) if parts else np.zeros((0, num_perm), dtype=np.uint32)
    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for band in range(bands):
        buckets = {}
        band_rows = sigs[:, band * rows : (band + 1) * rows]
        for i in range(len(texts)):
            key = band_rows[i].tobytes()
            j = buckets.get(key)
            if j is None:
                buckets[key] = i
            elif find(i) != find(j):
                similarity = float(np.mean(sigs[i] == sigs[j]))
                if similarity >= threshold:
                    parent[find(i)] = find(j)
    duplicates = set()
    seen = {}
    for i in range(len(texts)):
        root = find(i)
        if root in seen:
            duplicates.add(i)
        else:
            seen[root] = i
    return duplicates, threshold


def _split_of(text, val_fraction):
    if val_fraction <= 0:
        return "train"
    return "val" if (zlib.crc32(text.encode("utf-8")) % 1_000_000) < val_fraction * 1_000_000 else "train"


def build_records(spec, task, run_dir, cache_dir):
    data = spec.data
    out_dir = Path(run_dir) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    splits = []
    stats = {"sources": {}, "filtered": 0, "exact_duplicates": 0, "near_duplicates": 0}
    seen = set()
    for source in data.sources:
        path = source_path(source, cache_dir)
        kept = 0
        total = 0
        for row in iter_rows(source["format"], path):
            total += 1
            rec = task.record(row, spec)
            if rec is None or not (data.min_chars <= len(rec["text"]) <= data.max_chars):
                stats["filtered"] += 1
                continue
            fp = _fingerprint(rec["text"])
            if fp in seen:
                stats["exact_duplicates"] += 1
                continue
            seen.add(fp)
            records.append(rec)
            splits.append(source["split"] if source["split"] != "auto" else _split_of(rec["text"], data.val_fraction))
            kept += 1
            if data.max_records and len(records) >= data.max_records:
                break
            if total % 200000 == 0:
                print(f"{source['name']}: {total} rows read, {kept} kept", flush=True)
        stats["sources"][source["name"]] = {"rows": total, "kept": kept}
        if data.max_records and len(records) >= data.max_records:
            break
    del seen
    drop = set()
    if data.dedup == "minhash" and records:
        drop, threshold = minhash_duplicates(
            [r["text"] for r in records], data.num_perm, data.bands, data.shingle, data.seed, data.workers
        )
        stats["near_duplicates"] = len(drop)
        stats["minhash_threshold"] = threshold
    counts = {"train": 0, "val": 0}
    with ExitStack() as stack:
        files = {name: stack.enter_context(open(out_dir / f"{name}.jsonl", "w", encoding="utf-8")) for name in counts}
        for i, rec in enumerate(records):
            if i in drop:
                continue
            files[splits[i]].write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[splits[i]] += 1
    stats.update(counts)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"records: train {counts['train']} val {counts['val']}", flush=True)
    return stats


def iter_texts(path, field="text"):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)[field]


def iter_records(path, limit=0):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            yield json.loads(line)


def train_tokenizer(spec, run_dir):
    tok_dir = Path(run_dir) / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    texts = iter_texts(Path(run_dir) / "data" / "train.jsonl")
    tok = Tokenizer.train(texts, spec.tokenizer.vocab_size, spec.tokenizer.sample_bytes)
    tok.save(tok_dir / "tokenizer.json")
    print(f"tokenizer: {tok.vocab_size} tokens, {len(tok.merges)} merges", flush=True)
    return tok


def write_shards(spec, run_dir):
    run_dir = Path(run_dir)
    tok_path = run_dir / "tokenizer" / "tokenizer.json"
    tok = Tokenizer.load(tok_path)
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    meta = {"vocab_size": tok.vocab_size, "eos_id": tok.eos_id, "shards": {}}
    for split in ("train", "val"):
        texts = iter_texts(run_dir / "data" / f"{split}.jsonl")
        buffer = []
        count = 0
        index = 0
        names = []
        for ids in encode_many(str(tok_path), texts, spec.data.workers):
            buffer.extend(ids)
            buffer.append(tok.eos_id)
            count += 1
            if len(buffer) >= SHARD_TOKENS:
                name = f"{split}_{index:03d}.bin"
                np.array(buffer, dtype=np.uint16).tofile(shard_dir / name)
                names.append((name, len(buffer)))
                buffer = []
                index += 1
                print(f"{split}: {count} docs, shard {index} written", flush=True)
        if buffer or not names:
            name = f"{split}_{index:03d}.bin"
            np.array(buffer, dtype=np.uint16).tofile(shard_dir / name)
            names.append((name, len(buffer)))
        meta["shards"][split] = names
        meta[f"{split}_docs"] = count
        meta[f"{split}_tokens"] = sum(n for _, n in names)
    with open(shard_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"shards: train {meta['train_tokens']} tokens, val {meta['val_tokens']} tokens", flush=True)
    return meta
