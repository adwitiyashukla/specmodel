import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from specmodel.data import iter_records
from specmodel.model import generate
from specmodel.tokenizer import Tokenizer, chat_example_ids, chat_prompt_ids
from specmodel.train import Trainer, load_checkpoint, pick_dtype, setup_distributed, teardown_distributed, wrap_model


def stage_checkpoint(run_dir, order=("dpo", "sft", "pretrain")):
    for stage in order:
        path = Path(run_dir) / stage / "ckpt.pt"
        if path.exists():
            return stage, path
    raise FileNotFoundError(f"no checkpoint under {run_dir}")


def collate(examples, pad_id, device):
    length = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), length), pad_id, dtype=torch.long)
    y = torch.full((len(examples), length), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(examples):
        x[i, : len(ids)] = torch.from_numpy(ids)
        y[i, : len(labels)] = torch.from_numpy(labels)
    return x.to(device), y.to(device)


def shift(x, y):
    return x[:, :-1], y[:, 1:]


def load_examples(tok, spec, path, limit):
    system = spec.serve.system_prompt
    examples = []
    for rec in iter_records(path, limit):
        ids, labels = chat_example_ids(tok, system, rec["prompt"], rec["response"], spec.model.seq_len)
        examples.append((np.array(ids, dtype=np.int64), np.array(labels, dtype=np.int64)))
    return examples


def _epochs(examples, epochs, batch_size, rank, world, seed):
    rng = random.Random(seed)
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        order = order[rank::world]
        for i in range(0, len(order) - batch_size + 1, batch_size):
            yield [examples[j] for j in order[i : i + batch_size]]


def _run_stage(spec, run_dir, name, model, examples, section, loss_fn, batch_size, rank, world, device, extra):
    steps_per_epoch = len(examples) // (batch_size * world)
    total_steps = steps_per_epoch * section.epochs
    if total_steps == 0:
        raise ValueError(f"{name}: {len(examples)} examples is not enough for one batch")
    dtype = pick_dtype(spec.pretrain.dtype, device)
    trainer = Trainer(
        model,
        model,
        Path(run_dir) / name,
        section.lr,
        section.lr * 0.1,
        section.warmup_steps,
        spec.pretrain.weight_decay,
        spec.pretrain.grad_clip,
        dtype,
        device,
        rank,
        world,
        spec.pretrain.log_every,
    )
    wrapped = wrap_model(model, spec.pretrain.compile, device, world)
    trainer.model = wrapped
    if rank == 0:
        print(f"{name}: {len(examples)} examples, {total_steps} steps, {world} process(es)", flush=True)
    started = time.time()
    batches = _epochs(examples, section.epochs, batch_size, rank, world, spec.pretrain.seed)
    losses = []
    for step in range(total_steps):
        loss, lr, norm = trainer.train_step(step, total_steps, lambda b: loss_fn(wrapped, b), batches, 1)
        losses.append(loss)
        if (step + 1) % spec.pretrain.log_every == 0 or step == total_steps - 1:
            recent = sum(losses[-spec.pretrain.log_every :]) / len(losses[-spec.pretrain.log_every :])
            trainer.log(
                {"step": step + 1, "loss": recent, "lr": lr, "grad_norm": norm, "elapsed_s": time.time() - started}
            )
    trainer.save(total_steps)
    if rank == 0:
        summary = {
            "examples": len(examples),
            "steps": total_steps,
            "epochs": section.epochs,
            "final_loss": sum(losses[-20:]) / len(losses[-20:]),
            "train_seconds": time.time() - started,
        }
        summary.update(extra)
        with open(Path(run_dir) / name / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def sft(spec, run_dir):
    rank, world, device = setup_distributed()
    run_dir = Path(run_dir)
    tok = Tokenizer.load(run_dir / "tokenizer" / "tokenizer.json")
    model, _ = load_checkpoint(run_dir / "pretrain" / "ckpt.pt", device)
    examples = load_examples(tok, spec, run_dir / "data" / "train.jsonl", spec.sft.max_examples)
    pad_id = tok.pad_id

    def loss_fn(wrapped, batch):
        x, y = shift(*collate(batch, pad_id, device))
        return wrapped(x, targets=y)

    _run_stage(spec, run_dir, "sft", model, examples, spec.sft, loss_fn, spec.sft.batch_size, rank, world, device, {})
    teardown_distributed()


@torch.no_grad()
def mine_pairs(spec, run_dir, task):
    run_dir = Path(run_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.load(run_dir / "tokenizer" / "tokenizer.json")
    model, _ = load_checkpoint(run_dir / "sft" / "ckpt.pt", device)
    model.eval()
    dtype = pick_dtype(spec.pretrain.dtype, device)
    gen = spec.generation
    dpo = spec.dpo
    system = spec.serve.system_prompt
    max_prompt = spec.model.seq_len - gen.max_new_tokens
    out_dir = run_dir / "dpo"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    stats = {"prompts": 0, "samples": 0, "samples_passed": 0, "on_policy_chosen": 0}
    batch_prompts = max(1, spec.eval.batch_size // dpo.candidates)
    records = iter_records(run_dir / "data" / "train.jsonl", spec.sft.max_examples)
    started = time.time()
    while len(pairs) < dpo.pairs:
        batch = []
        for rec in records:
            batch.append(rec)
            if len(batch) == batch_prompts:
                break
        if not batch:
            break
        prompts = [
            chat_prompt_ids(tok, system, rec["prompt"], max_prompt) for rec in batch for _ in range(dpo.candidates)
        ]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            outputs = generate(
                model,
                prompts,
                tok.eos_id,
                tok.pad_id,
                gen.max_new_tokens,
                dpo.temperature,
                gen.top_k,
                gen.top_p,
                gen.repetition_penalty,
            )
        for i, rec in enumerate(batch):
            stats["prompts"] += 1
            texts = [tok.decode(o) for o in outputs[i * dpo.candidates : (i + 1) * dpo.candidates]]
            scored = [(t, task.passed(task.score(rec, t, spec))) for t in texts]
            stats["samples"] += len(scored)
            stats["samples_passed"] += sum(1 for _, ok in scored if ok)
            failed = [t for t, ok in scored if not ok]
            passed = [t for t, ok in scored if ok]
            if not failed:
                continue
            if passed:
                chosen = passed[0]
                stats["on_policy_chosen"] += 1
            else:
                chosen = rec["response"]
            pairs.append({"prompt": rec["prompt"], "chosen": chosen, "rejected": failed[0]})
            if len(pairs) >= dpo.pairs:
                break
        if stats["prompts"] % (batch_prompts * 20) == 0:
            print(f"mined {len(pairs)} pairs from {stats['prompts']} prompts, {time.time() - started:.0f}s", flush=True)
    stats["pairs"] = len(pairs)
    stats["sample_pass_rate"] = stats["samples_passed"] / max(stats["samples"], 1)
    stats["mine_seconds"] = time.time() - started
    with open(out_dir / "pairs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs)
    with open(out_dir / "mining.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"pairs: {len(pairs)}, sample pass rate {stats['sample_pass_rate']:.3f}", flush=True)
    return stats


def sequence_logprobs(model, x, y):
    x, y = shift(x, y)
    return model(x, targets=y, per_token=True).sum(dim=-1)


def dpo(spec, run_dir):
    rank, world, device = setup_distributed()
    run_dir = Path(run_dir)
    tok = Tokenizer.load(run_dir / "tokenizer" / "tokenizer.json")
    model, _ = load_checkpoint(run_dir / "sft" / "ckpt.pt", device)
    reference = copy.deepcopy(model).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    system = spec.serve.system_prompt
    examples = []
    for pair in iter_records(run_dir / "dpo" / "pairs.jsonl"):
        chosen = chat_example_ids(tok, system, pair["prompt"], pair["chosen"], spec.model.seq_len)
        rejected = chat_example_ids(tok, system, pair["prompt"], pair["rejected"], spec.model.seq_len)
        examples.append(tuple(np.array(v, dtype=np.int64) for v in chosen + rejected))
    pad_id = tok.pad_id
    beta = spec.dpo.beta
    tracker = {"margin": [], "accuracy": []}

    def loss_fn(wrapped, batch):
        chosen = collate([(c, cl) for c, cl, _, _ in batch], pad_id, device)
        rejected = collate([(r, rl) for _, _, r, rl in batch], pad_id, device)
        policy_c = sequence_logprobs(wrapped, *chosen)
        policy_r = sequence_logprobs(wrapped, *rejected)
        with torch.no_grad():
            ref_c = sequence_logprobs(reference, *chosen)
            ref_r = sequence_logprobs(reference, *rejected)
        logits = beta * ((policy_c - ref_c) - (policy_r - ref_r))
        tracker["margin"].append(float(logits.mean().item()))
        tracker["accuracy"].append(float((logits > 0).float().mean().item()))
        return -F.logsigmoid(logits).mean()

    def summary():
        tail = max(1, len(tracker["margin"]) // 10)
        return {
            "reward_margin": sum(tracker["margin"][-tail:]) / tail,
            "reward_accuracy": sum(tracker["accuracy"][-tail:]) / tail,
        }

    _run_stage(spec, run_dir, "dpo", model, examples, spec.dpo, loss_fn, spec.dpo.batch_size, rank, world, device, {})
    if rank == 0:
        path = run_dir / "dpo" / "summary.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data.update(summary())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    teardown_distributed()
