import tomllib
from pathlib import Path


class SpecError(ValueError):
    pass


class Section(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(key) from None
        return value


TASKS = ("story", "sql")
FORMATS = ("tinystories", "parquet", "jsonl", "text")
DEDUP_METHODS = ("exact", "minhash")
DTYPES = ("auto", "float32", "float16", "bfloat16")
QUANTIZE = ("none", "int8")
SPLITS = ("auto", "train", "val")

FIELDS = {
    ("client", "name"): (str, None, None),
    ("client", "task"): (str, None, None),
    ("client", "description"): (str, None, None),
    ("data", "min_chars"): (int, 1, None),
    ("data", "max_chars"): (int, 1, None),
    ("data", "val_fraction"): (float, 0.0, 0.5),
    ("data", "max_records"): (int, 0, None),
    ("data", "seed"): (int, 0, None),
    ("data", "dedup"): (str, None, None),
    ("data", "num_perm"): (int, 8, 512),
    ("data", "bands"): (int, 1, 512),
    ("data", "shingle"): (int, 1, 20),
    ("data", "workers"): (int, 1, 64),
    ("tokenizer", "vocab_size"): (int, 300, 65535),
    ("tokenizer", "sample_bytes"): (int, 1000, None),
    ("model", "dim"): (int, 8, None),
    ("model", "n_layers"): (int, 1, None),
    ("model", "n_heads"): (int, 1, None),
    ("model", "n_kv_heads"): (int, 1, None),
    ("model", "seq_len"): (int, 8, None),
    ("model", "rope_theta"): (float, 1.0, None),
    ("pretrain", "tokens"): (int, 1, None),
    ("pretrain", "batch_tokens"): (int, 1, None),
    ("pretrain", "micro_batch"): (int, 1, None),
    ("pretrain", "lr"): (float, 0.0, None),
    ("pretrain", "min_lr"): (float, 0.0, None),
    ("pretrain", "warmup_steps"): (int, 0, None),
    ("pretrain", "weight_decay"): (float, 0.0, None),
    ("pretrain", "grad_clip"): (float, 0.0, None),
    ("pretrain", "eval_every"): (int, 1, None),
    ("pretrain", "eval_batches"): (int, 1, None),
    ("pretrain", "checkpoint_every"): (int, 1, None),
    ("pretrain", "log_every"): (int, 1, None),
    ("pretrain", "dtype"): (str, None, None),
    ("pretrain", "compile"): (bool, None, None),
    ("pretrain", "seed"): (int, 0, None),
    ("sft", "epochs"): (int, 1, None),
    ("sft", "lr"): (float, 0.0, None),
    ("sft", "batch_size"): (int, 1, None),
    ("sft", "max_examples"): (int, 0, None),
    ("sft", "warmup_steps"): (int, 0, None),
    ("dpo", "beta"): (float, 0.0, None),
    ("dpo", "lr"): (float, 0.0, None),
    ("dpo", "pairs"): (int, 1, None),
    ("dpo", "epochs"): (int, 1, None),
    ("dpo", "batch_size"): (int, 1, None),
    ("dpo", "candidates"): (int, 1, 16),
    ("dpo", "temperature"): (float, 0.0, None),
    ("dpo", "warmup_steps"): (int, 0, None),
    ("generation", "max_new_tokens"): (int, 1, None),
    ("generation", "temperature"): (float, 0.0, None),
    ("generation", "top_p"): (float, 0.0, 1.0),
    ("generation", "top_k"): (int, 0, None),
    ("generation", "repetition_penalty"): (float, 1.0, None),
    ("eval", "samples"): (int, 1, None),
    ("eval", "batch_size"): (int, 1, None),
    ("eval", "perplexity_batches"): (int, 1, None),
    ("export", "quantize"): (str, None, None),
    ("serve", "system_prompt"): (str, None, None),
    ("serve", "max_batch"): (int, 1, None),
    ("serve", "batch_wait_ms"): (int, 0, None),
}

CHOICES = {
    ("client", "task"): TASKS,
    ("data", "dedup"): DEDUP_METHODS,
    ("pretrain", "dtype"): DTYPES,
    ("export", "quantize"): QUANTIZE,
}

TASK_FIELDS = {
    "story": {
        "min_words": (int, 1, None),
        "max_words": (int, 1, None),
        "banned_words": (list, None, None),
    },
    "sql": {
        "timeout_ms": (int, 1, None),
        "max_rows": (int, 1, None),
    },
}


def _wrap(value):
    if isinstance(value, dict):
        return Section({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _check(path, value, kind, low, high):
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if kind is int and isinstance(value, bool):
        raise SpecError(f"{path} must be an integer")
    if not isinstance(value, kind):
        raise SpecError(f"{path} must be {kind.__name__}")
    if low is not None and value < low:
        raise SpecError(f"{path} must be at least {low}")
    if high is not None and value > high:
        raise SpecError(f"{path} must be at most {high}")
    return value


def validate(raw):
    for (section, key), (kind, low, high) in FIELDS.items():
        if section not in raw:
            raise SpecError(f"missing section [{section}]")
        if key not in raw[section]:
            raise SpecError(f"missing {section}.{key}")
        raw[section][key] = _check(f"{section}.{key}", raw[section][key], kind, low, high)
        if (section, key) in CHOICES and raw[section][key] not in CHOICES[(section, key)]:
            raise SpecError(f"{section}.{key} must be one of {CHOICES[(section, key)]}")
    sources = raw["data"].get("sources")
    if not isinstance(sources, list) or not sources:
        raise SpecError("data.sources must list at least one source")
    for i, source in enumerate(sources):
        for key in ("name", "format"):
            if not isinstance(source.get(key), str):
                raise SpecError(f"data.sources[{i}].{key} must be a string")
        if source["format"] not in FORMATS:
            raise SpecError(f"data.sources[{i}].format must be one of {FORMATS}")
        if not isinstance(source.get("url"), str) and not isinstance(source.get("path"), str):
            raise SpecError(f"data.sources[{i}] needs url or path")
        source.setdefault("split", "auto")
        if source["split"] not in SPLITS:
            raise SpecError(f"data.sources[{i}].split must be one of {SPLITS}")
    if raw["data"]["min_chars"] > raw["data"]["max_chars"]:
        raise SpecError("data.min_chars must not exceed data.max_chars")
    if raw["data"]["num_perm"] % raw["data"]["bands"] != 0:
        raise SpecError("data.num_perm must be divisible by data.bands")
    model = raw["model"]
    if model["dim"] % model["n_heads"] != 0:
        raise SpecError("model.dim must be divisible by model.n_heads")
    if model["n_heads"] % model["n_kv_heads"] != 0:
        raise SpecError("model.n_heads must be divisible by model.n_kv_heads")
    if (model["dim"] // model["n_heads"]) % 2 != 0:
        raise SpecError("head dimension must be even for rotary embeddings")
    pre = raw["pretrain"]
    if pre["min_lr"] > pre["lr"]:
        raise SpecError("pretrain.min_lr must not exceed pretrain.lr")
    if pre["batch_tokens"] < pre["micro_batch"] * model["seq_len"]:
        raise SpecError("pretrain.batch_tokens must cover at least one micro batch")
    gates = raw["eval"].get("gates", [])
    if not isinstance(gates, list):
        raise SpecError("eval.gates must be a list of tables")
    for i, gate in enumerate(gates):
        if not isinstance(gate.get("metric"), str):
            raise SpecError(f"eval.gates[{i}].metric must be a string")
        if "min" not in gate and "max" not in gate:
            raise SpecError(f"eval.gates[{i}] needs min or max")
        for bound in ("min", "max"):
            if bound in gate:
                gate[bound] = _check(f"eval.gates[{i}].{bound}", gate[bound], float, None, None)
    raw["eval"]["gates"] = gates
    stop = raw["serve"].get("stop", [])
    if not isinstance(stop, list) or not all(isinstance(x, str) and x for x in stop):
        raise SpecError("serve.stop must be a list of non empty strings")
    raw["serve"]["stop"] = stop
    raw["serve"]["example"] = _check("serve.example", raw["serve"].get("example", ""), str, None, None)
    task = raw["client"]["task"]
    if "task" not in raw:
        raise SpecError("missing section [task]")
    for key, (kind, low, high) in TASK_FIELDS[task].items():
        if key not in raw["task"]:
            raise SpecError(f"missing task.{key}")
        raw["task"][key] = _check(f"task.{key}", raw["task"][key], kind, low, high)
    if task == "story":
        if raw["task"]["min_words"] > raw["task"]["max_words"]:
            raise SpecError("task.min_words must not exceed task.max_words")
        if not all(isinstance(w, str) for w in raw["task"]["banned_words"]):
            raise SpecError("task.banned_words must be strings")
    return raw


def load_spec(path):
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    spec = _wrap(validate(raw))
    spec["path"] = str(path.resolve())
    return spec
