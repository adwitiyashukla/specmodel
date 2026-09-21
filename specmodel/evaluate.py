import json
import math
import time
from pathlib import Path

import torch

from specmodel.data import iter_records
from specmodel.export import load_export, write_model_card
from specmodel.model import generate
from specmodel.tokenizer import Tokenizer, chat_prompt_ids
from specmodel.train import ShardSampler, load_checkpoint, pick_dtype


def check_gates(metrics, gates):
    results = []
    for gate in gates:
        value = metrics.get(gate["metric"])
        ok = value is not None
        if ok and "min" in gate:
            ok = value >= gate["min"]
        if ok and "max" in gate:
            ok = value <= gate["max"]
        row = {"metric": gate["metric"], "value": value if value is not None else float("nan"), "ok": bool(ok)}
        for bound in ("min", "max"):
            if bound in gate:
                row[bound] = gate[bound]
        results.append(row)
    return results


def load_stage(spec, run_dir, stage, device):
    run_dir = Path(run_dir)
    if stage == "export":
        model, tok, _ = load_export(run_dir / "export", device)
        return model, tok
    model, _ = load_checkpoint(run_dir / stage / "ckpt.pt", device)
    tok = Tokenizer.load(run_dir / "tokenizer" / "tokenizer.json")
    return model.eval(), tok


@torch.no_grad()
def perplexity(model, sampler, batches, device, dtype):
    total = 0.0
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        for _ in range(batches):
            x, y = sampler.batch()
            total += float(model(x.to(device), targets=y.to(device)).item())
    return math.exp(total / batches)


@torch.no_grad()
def evaluate(spec, run_dir, stage, task):
    run_dir = Path(run_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype(spec.pretrain.dtype, device)
    model, tok = load_stage(spec, run_dir, stage, device)
    gen = spec.generation
    sampler = ShardSampler(run_dir / "shards", "val", spec.model.seq_len, spec.pretrain.micro_batch, spec.pretrain.seed)
    ppl = perplexity(model, sampler, spec.eval.perplexity_batches, device, dtype)
    records = list(iter_records(run_dir / "data" / "val.jsonl", spec.eval.samples))
    system = spec.serve.system_prompt
    max_prompt = spec.model.seq_len - gen.max_new_tokens
    rows = []
    samples = []
    generated_tokens = 0
    started = time.time()
    for i in range(0, len(records), spec.eval.batch_size):
        batch = records[i : i + spec.eval.batch_size]
        prompts = [chat_prompt_ids(tok, system, rec["prompt"], max_prompt) for rec in batch]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            outputs = generate(
                model,
                prompts,
                tok.eos_id,
                tok.pad_id,
                gen.max_new_tokens,
                gen.temperature,
                gen.top_k,
                gen.top_p,
                gen.repetition_penalty,
            )
        for rec, out in zip(batch, outputs):
            generated_tokens += len(out)
            text = tok.decode(out)
            scores = task.score(rec, text, spec)
            rows.append(scores)
            if len(samples) < 5:
                samples.append({"prompt": rec["prompt"], "output": task.finalize(text), "scores": scores})
        print(f"eval {stage}: {len(rows)}/{len(records)} samples, {time.time() - started:.0f}s", flush=True)
    elapsed = time.time() - started
    metrics = task.summarize(rows)
    gates = check_gates(metrics, spec.eval.gates)
    report = {
        "stage": stage,
        "params": model.num_params(),
        "perplexity": ppl,
        "metrics": metrics,
        "gates": gates,
        "gates_passed": all(g["ok"] for g in gates),
        "samples_evaluated": len(rows),
        "generation_tokens_per_s": generated_tokens / max(elapsed, 1e-6),
        "device": device.type,
        "samples": samples,
    }
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stage}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if stage == "export" and (run_dir / "export").exists():
        write_model_card(spec, run_dir)
    summary = " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    print(f"eval {stage}: ppl={ppl:.2f} {summary} gates={'pass' if report['gates_passed'] else 'FAIL'}", flush=True)
    return report
