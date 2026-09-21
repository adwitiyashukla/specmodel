import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from specmodel.model import ModelConfig, Transformer


def setup_distributed():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        return rank, world, device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def teardown_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def pick_dtype(name, device):
    if device.type != "cuda":
        return torch.float32
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def all_reduce_mean(value, world):
    if world == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(tensor)
    return float(tensor.item() / world)


class ShardSampler:
    def __init__(self, shard_dir, split, seq_len, batch_size, seed, rank=0, world=1):
        shard_dir = Path(shard_dir)
        with open(shard_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.shards = [np.memmap(shard_dir / name, dtype=np.uint16, mode="r") for name, _ in meta["shards"][split]]
        self.shards = [s for s in self.shards if len(s) > seq_len + 1]
        if not self.shards:
            raise ValueError(f"{split} split has fewer than {seq_len + 2} tokens")
        sizes = np.array([len(s) for s in self.shards], dtype=np.float64)
        self.weights = sizes / sizes.sum()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed * 7919 + rank)
        self.world = world

    def batch(self):
        xs = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
        ys = np.empty((self.batch_size, self.seq_len), dtype=np.int64)
        for i in range(self.batch_size):
            shard = self.shards[self.rng.choice(len(self.shards), p=self.weights)]
            start = int(self.rng.integers(0, len(shard) - self.seq_len - 1))
            window = shard[start : start + self.seq_len + 1].astype(np.int64)
            xs[i] = window[:-1]
            ys[i] = window[1:]
        return torch.from_numpy(xs), torch.from_numpy(ys)


class Trainer:
    def __init__(
        self,
        model,
        raw_model,
        out_dir,
        lr,
        min_lr,
        warmup_steps,
        weight_decay,
        grad_clip,
        dtype,
        device,
        rank,
        world,
        log_every,
    ):
        self.model = model
        self.raw_model = raw_model
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.dtype = dtype
        self.device = device
        self.rank = rank
        self.world = world
        self.log_every = log_every
        decay = [p for p in raw_model.parameters() if p.dim() >= 2]
        no_decay = [p for p in raw_model.parameters() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        self.optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=device.type == "cuda")
        self.scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
        self.log_path = self.out_dir / "log.csv"
        self.started = time.time()

    def lr_at(self, step, total_steps):
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        if total_steps <= self.warmup_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / max(total_steps - self.warmup_steps, 1)
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32)

    def train_step(self, step, total_steps, loss_fn, batches, accumulation):
        lr = self.lr_at(step, total_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.model.train()
        total = 0.0
        for micro in range(accumulation):
            batch = next(batches)
            sync = micro == accumulation - 1
            context = self.model.no_sync() if (self.world > 1 and not sync) else nullcontext()
            with context, self.autocast():
                loss = loss_fn(batch) / accumulation
            self.scaler.scale(loss).backward()
            total += float(loss.item())
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return total, lr, float(norm)

    def log(self, row):
        if self.rank != 0:
            return
        new = not self.log_path.exists()
        with open(self.log_path, "a", encoding="utf-8") as f:
            if new:
                f.write(",".join(row.keys()) + "\n")
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row.values()) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    def save(self, step, extra=None):
        if self.rank != 0:
            return
        state = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": step,
            "config": self.raw_model.config.to_dict(),
            "extra": extra or {},
        }
        tmp = self.out_dir / "ckpt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.out_dir / "ckpt.pt")

    def resume(self):
        path = self.out_dir / "ckpt.pt"
        if not path.exists():
            return 0
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        if self.rank == 0:
            print(f"resumed from step {state['step']}", flush=True)
        return int(state["step"])


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**state["config"])
    model = Transformer(config)
    model.load_state_dict(state["model"])
    return model.to(device), state


def wrap_model(model, spec_compile, device, world):
    wrapped = model
    if world > 1:
        ids = [device.index] if device.type == "cuda" else None
        wrapped = DistributedDataParallel(wrapped, device_ids=ids)
    if spec_compile:
        wrapped = torch.compile(wrapped)
    return wrapped


@torch.no_grad()
def val_loss(model, sampler, batches, device, autocast):
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = sampler.batch()
        with autocast:
            total += float(model(x.to(device), targets=y.to(device)).item())
    model.train()
    return total / batches


def pretrain(spec, run_dir):
    rank, world, device = setup_distributed()
    run_dir = Path(run_dir)
    with open(run_dir / "shards" / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    pre = spec.pretrain
    torch.manual_seed(pre.seed + rank)
    config = ModelConfig.from_spec(spec, meta["vocab_size"])
    model = Transformer(config).to(device)
    seq_len = config.seq_len
    accumulation = max(1, pre.batch_tokens // (pre.micro_batch * seq_len * world))
    tokens_per_step = accumulation * pre.micro_batch * seq_len * world
    total_steps = max(1, math.ceil(pre.tokens / tokens_per_step))
    dtype = pick_dtype(pre.dtype, device)
    trainer = Trainer(
        model,
        model,
        run_dir / "pretrain",
        pre.lr,
        pre.min_lr,
        pre.warmup_steps,
        pre.weight_decay,
        pre.grad_clip,
        dtype,
        device,
        rank,
        world,
        pre.log_every,
    )
    start = trainer.resume()
    if start >= total_steps:
        if rank == 0:
            print(f"pretrain already complete at step {start}", flush=True)
        teardown_distributed()
        return
    wrapped = wrap_model(model, pre.compile, device, world)
    trainer.model = wrapped
    train_sampler = ShardSampler(run_dir / "shards", "train", seq_len, pre.micro_batch, pre.seed + start, rank, world)
    val_sampler = ShardSampler(run_dir / "shards", "val", seq_len, pre.micro_batch, pre.seed + 1, rank, world)
    if rank == 0:
        print(
            f"model {model.num_params() / 1e6:.1f}M params, {total_steps} steps of {tokens_per_step} tokens, "
            f"{world} process(es), dtype {dtype}",
            flush=True,
        )

    def batches():
        while True:
            x, y = train_sampler.batch()
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def loss_fn(batch):
        x, y = batch
        return wrapped(x, targets=y)

    stream = batches()
    started = time.time()
    seen = 0
    best_val = None
    for step in range(start, total_steps):
        loss, lr, norm = trainer.train_step(step, total_steps, loss_fn, stream, accumulation)
        seen += tokens_per_step
        if (step + 1) % pre.log_every == 0 or step == total_steps - 1:
            elapsed = time.time() - started
            trainer.log(
                {
                    "step": step + 1,
                    "loss": loss,
                    "lr": lr,
                    "grad_norm": norm,
                    "tokens_per_s": seen / max(elapsed, 1e-6),
                    "elapsed_s": elapsed,
                }
            )
        if (step + 1) % pre.eval_every == 0 or step == total_steps - 1:
            v = all_reduce_mean(val_loss(wrapped, val_sampler, pre.eval_batches, device, trainer.autocast()), world)
            best_val = v if best_val is None else min(best_val, v)
            trainer.log({"step": step + 1, "val_loss": v, "val_ppl": math.exp(v)})
        if (step + 1) % pre.checkpoint_every == 0 or step == total_steps - 1:
            trainer.save(step + 1, {"val_loss": best_val})
    if rank == 0:
        summary = {
            "params": model.num_params(),
            "steps": total_steps,
            "tokens": total_steps * tokens_per_step,
            "tokens_per_step": tokens_per_step,
            "world": world,
            "dtype": str(dtype),
            "val_loss": best_val,
            "train_seconds": time.time() - started,
        }
        with open(run_dir / "pretrain" / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    teardown_distributed()
