import json

from specmodel.cli import main
from specmodel.evaluate import check_gates


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_sql_build_produces_every_artifact(sql_run):
    for name in (
        "data/stats.json",
        "tokenizer/tokenizer.json",
        "shards/meta.json",
        "pretrain/ckpt.pt",
        "pretrain/summary.json",
        "sft/ckpt.pt",
        "dpo/pairs.jsonl",
        "dpo/mining.json",
        "dpo/ckpt.pt",
        "dpo/summary.json",
        "export/model.safetensors",
        "export/config.json",
        "export/tokenizer.json",
        "export/spec.toml",
        "export/model_card.md",
        "eval/pretrain.json",
        "eval/sft.json",
        "eval/dpo.json",
        "eval/export.json",
    ):
        assert (sql_run / name).exists(), name
    pretrain = _json(sql_run / "pretrain" / "summary.json")
    assert pretrain["steps"] == 20 and pretrain["tokens"] == 20 * 2048
    log = (sql_run / "pretrain" / "log.csv").read_text(encoding="utf-8").splitlines()
    assert log[0].startswith("step,loss,lr")
    first, last = float(log[1].split(",")[1]), float(log[-2].split(",")[1])
    assert last < first
    report = _json(sql_run / "eval" / "export.json")
    assert set(report["metrics"]) == {
        "valid_rate",
        "exact_match",
        "gold_executable_rate",
        "execution_accuracy",
        "pass_rate",
    }
    assert report["samples_evaluated"] == 8 and len(report["samples"]) == 5
    assert report["perplexity"] > 1.0
    assert [g["metric"] for g in report["gates"]] == ["execution_accuracy", "valid_rate"]
    dpo = _json(sql_run / "dpo" / "summary.json")
    assert "reward_accuracy" in dpo and dpo["steps"] == 2
    card = (sql_run / "export" / "model_card.md").read_text(encoding="utf-8")
    assert "| export |" in card and "## Gates" in card and "execution_accuracy" in card
    config = _json(sql_run / "export" / "config.json")
    assert config["stage"] == "dpo" and config["quantize"] == "int8" and config["model"]["dim"] == 64


def test_build_is_resumable_and_skips_finished_stages(sql_run, sql_spec, capsys):
    try:
        main(
            ["--runs", str(sql_run.parent), "--cache", str(sql_run.parent / "cache"), "build", "--spec", str(sql_spec)]
        )
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert out.count(": done") == 8 and "running" not in out
    assert "build passed" in out or "build failed" in out


def test_story_pipeline_mines_pairs_and_scores_constraints(story_run):
    mining = _json(story_run / "dpo" / "mining.json")
    assert mining["pairs"] == 4 and mining["samples"] >= 8
    pairs = [json.loads(line) for line in (story_run / "dpo" / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(pairs) == 4 and all(set(p) == {"prompt", "chosen", "rejected"} for p in pairs)
    report = _json(story_run / "eval" / "export.json")
    assert "pass_rate" in report["metrics"] and "banned_rate" in report["metrics"]
    assert report["metrics"]["banned_rate"] == 0.0
    sft = _json(story_run / "eval" / "sft.json")
    assert sft["stage"] == "sft"
    card = (story_run / "export" / "model_card.md").read_text(encoding="utf-8")
    assert "| sft |" in card and "| export |" in card


def test_gate_checks():
    gates = [{"metric": "a", "min": 0.5}, {"metric": "b", "max": 0.1}, {"metric": "missing", "min": 0.0}]
    results = check_gates({"a": 0.6, "b": 0.2}, gates)
    assert [r["ok"] for r in results] == [True, False, False]
    assert results[0]["min"] == 0.5 and results[1]["max"] == 0.1
