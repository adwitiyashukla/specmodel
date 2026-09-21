from pathlib import Path

import pytest
from synthetic import small_spec, write_sql_parquet, write_story_tar

from specmodel.cli import main

ROOT = Path(__file__).resolve().parent.parent
TINY_MODEL = {"dim": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "seq_len": 128}
TINY_PRETRAIN = {
    "tokens": 40000,
    "batch_tokens": 2048,
    "micro_batch": 8,
    "warmup_steps": 2,
    "eval_every": 10,
    "checkpoint_every": 10,
    "log_every": 5,
    "eval_batches": 2,
    "lr": 0.003,
    "min_lr": 0.0003,
}


@pytest.fixture(scope="session")
def story_spec(tmp_path_factory):
    root = tmp_path_factory.mktemp("story")
    write_story_tar(root / "stories.tar.gz", 400)
    return small_spec(
        ROOT / "clients" / "storyteller" / "spec.toml",
        root / "spec.toml",
        [{"name": "tinystories", "path": str(root / "stories.tar.gz"), "format": "tinystories", "split": "auto"}],
        {
            "data": {"val_fraction": 0.1, "workers": 1},
            "tokenizer": {"vocab_size": 400, "sample_bytes": 100000},
            "model": TINY_MODEL,
            "pretrain": TINY_PRETRAIN,
            "sft": {"max_examples": 64, "batch_size": 8, "warmup_steps": 1},
            "dpo": {"pairs": 4, "batch_size": 2, "candidates": 2, "warmup_steps": 1},
            "generation": {"max_new_tokens": 24},
            "eval": {"samples": 8, "batch_size": 8, "perplexity_batches": 2},
        },
    )


@pytest.fixture(scope="session")
def sql_spec(tmp_path_factory):
    root = tmp_path_factory.mktemp("sql")
    write_sql_parquet(root / "train.parquet", 300, seed=1)
    write_sql_parquet(root / "test.parquet", 40, seed=2)
    return small_spec(
        ROOT / "clients" / "sqlbot" / "spec.toml",
        root / "spec.toml",
        [
            {"name": "train", "path": str(root / "train.parquet"), "format": "parquet", "split": "train"},
            {"name": "test", "path": str(root / "test.parquet"), "format": "parquet", "split": "val"},
        ],
        {
            "data": {"workers": 1},
            "tokenizer": {"vocab_size": 400, "sample_bytes": 100000},
            "model": TINY_MODEL,
            "pretrain": TINY_PRETRAIN,
            "sft": {"max_examples": 0, "batch_size": 8, "warmup_steps": 1, "epochs": 1},
            "dpo": {"pairs": 4, "batch_size": 2, "candidates": 2, "warmup_steps": 1},
            "generation": {"max_new_tokens": 24},
            "eval": {"samples": 8, "batch_size": 8, "perplexity_batches": 2},
        },
    )


@pytest.fixture(scope="session")
def sql_run(sql_spec, tmp_path_factory):
    root = tmp_path_factory.mktemp("sqlrun")
    try:
        main(["--runs", str(root), "--cache", str(root / "cache"), "build", "--spec", str(sql_spec)])
    except SystemExit:
        pass
    return root / "sqlbot"


@pytest.fixture(scope="session")
def story_run(story_spec, tmp_path_factory):
    root = tmp_path_factory.mktemp("storyrun")
    args = ["--runs", str(root), "--cache", str(root / "cache")]
    for stage in ("data", "tokenizer", "tokenize", "pretrain", "sft", "mine", "dpo", "export"):
        main(args + [stage, "--spec", str(story_spec)])
    main(args + ["eval", "--spec", str(story_spec), "--stage", "sft", "export"])
    return root / "storyteller"
