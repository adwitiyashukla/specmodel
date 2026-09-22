import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from specmodel.model import ModelConfig, Transformer
from specmodel.spec import load_spec
from specmodel.tokenizer import Tokenizer

QUANT_SUFFIXES = ("wq.weight", "wk.weight", "wv.weight", "wo.weight", "w1.weight", "w2.weight", "w3.weight")
STAGES = ("pretrain", "sft", "dpo")


def quantize_int8(weight):
    scale = weight.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(weight / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.float()


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pick_stage(spec, run_dir):
    run_dir = Path(run_dir)
    gate = spec.eval.gates[0] if spec.eval.gates else None
    latest = None
    scored = []
    for stage in STAGES:
        path = run_dir / stage / "ckpt.pt"
        if not path.exists():
            continue
        latest = (stage, path, "latest")
        report = _read_json(run_dir / "eval" / f"{stage}.json")
        if gate and report and gate["metric"] in report["metrics"]:
            value = report["metrics"][gate["metric"]]
            scored.append((value if "min" in gate else -value, len(scored), stage, path))
    if latest is None:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    if not scored:
        return latest
    _, _, stage, path = max(scored)
    return stage, path, gate["metric"]


def export(spec, run_dir):
    run_dir = Path(run_dir)
    stage, path, picked_by = pick_stage(spec, run_dir)
    state = torch.load(path, map_location="cpu", weights_only=False)
    out = run_dir / "export"
    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, t in state["model"].items():
        if name == "lm_head.weight":
            continue
        if spec.export.quantize == "int8" and name.endswith(QUANT_SUFFIXES):
            q, s = quantize_int8(t.float())
            tensors[name] = q.contiguous()
            tensors[name + ".scale"] = s.contiguous()
        else:
            tensors[name] = t.to(torch.float16).contiguous()
    save_file(tensors, out / "model.safetensors", metadata={"quantize": spec.export.quantize, "stage": stage})
    config = {
        "model": state["config"],
        "quantize": spec.export.quantize,
        "stage": stage,
        "picked_by": picked_by,
        "client": spec.client.name,
        "task": spec.client.task,
    }
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    shutil.copy(run_dir / "tokenizer" / "tokenizer.json", out / "tokenizer.json")
    shutil.copy(spec.path, out / "spec.toml")
    write_model_card(spec, run_dir)
    size = (out / "model.safetensors").stat().st_size
    print(f"export: {stage} checkpoint picked by {picked_by}, {spec.export.quantize}, {size / 1e6:.1f} MB", flush=True)
    return out


def load_export(export_dir, device):
    export_dir = Path(export_dir)
    with open(export_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    tensors = load_file(str(export_dir / "model.safetensors"))
    state = {}
    for name, t in tensors.items():
        if name.endswith(".scale"):
            continue
        if t.dtype == torch.int8:
            state[name] = t.float() * tensors[name + ".scale"][:, None]
        else:
            state[name] = t.float()
    state["lm_head.weight"] = state["tok_emb.weight"]
    model = Transformer(ModelConfig(**config["model"]))
    model.load_state_dict(state)
    model.to(device).eval()
    tok = Tokenizer.load(export_dir / "tokenizer.json")
    spec = load_spec(export_dir / "spec.toml")
    return model, tok, spec


def write_model_card(spec, run_dir):
    run_dir = Path(run_dir)
    out = run_dir / "export"
    pretrain = _read_json(run_dir / "pretrain" / "summary.json") or {}
    config = _read_json(out / "config.json") or {}
    reports = {s: _read_json(run_dir / "eval" / f"{s}.json") for s in (*STAGES, "export")}
    reports = {s: r for s, r in reports.items() if r}
    m = spec.model
    lines = [f"# {spec.client.name}", "", spec.client.description, "", "## Model", ""]
    params = pretrain.get("params")
    lines.append(
        f"Decoder only transformer trained from scratch: {m.n_layers} layers, dim {m.dim}, {m.n_heads} heads "
        f"with {m.n_kv_heads} key value heads, context {m.seq_len}, vocabulary {spec.tokenizer.vocab_size} "
        f"(byte level BPE trained on the client corpus)" + (f", {params / 1e6:.1f}M parameters." if params else ".")
    )
    if pretrain:
        lines.append(
            f"Pretrained on {pretrain.get('tokens', 0) / 1e6:.0f}M tokens in {pretrain.get('steps')} steps "
            f"({pretrain.get('train_seconds', 0) / 60:.0f} minutes, {pretrain.get('world')} process(es), "
            f"{pretrain.get('dtype')})."
        )
    picked_by = config.get("picked_by", "latest")
    how = "the latest stage" if picked_by == "latest" else f"the stage with the best {picked_by}"
    lines.append(
        f"Post training: supervised fine tuning, then DPO on preference pairs mined from the model's own "
        f"failed samples. Export: the {config.get('stage', 'final')} checkpoint ({how}), "
        f"{spec.export.quantize} weights in safetensors."
    )
    if reports:
        keys = []
        for r in reports.values():
            for k in r["metrics"]:
                if k not in keys:
                    keys.append(k)
        lines += [
            "",
            "## Results",
            "",
            "| stage | perplexity | " + " | ".join(keys) + " |",
            "|---|---|" + "---|" * len(keys),
        ]
        for stage, r in reports.items():
            cells = [f"{r['metrics'].get(k, float('nan')):.3f}" for k in keys]
            lines.append(f"| {stage} | {r['perplexity']:.2f} | " + " | ".join(cells) + " |")
        final = reports.get("export") or list(reports.values())[-1]
        lines += ["", "## Gates", "", "| metric | bound | value | passed |", "|---|---|---|---|"]
        for g in final["gates"]:
            bound = f"min {g['min']}" if "min" in g else f"max {g['max']}"
            lines.append(f"| {g['metric']} | {bound} | {g['value']:.3f} | {'yes' if g['ok'] else 'no'} |")
        lines += ["", "## Samples", ""]
        for s in final["samples"][:3]:
            lines += ["Prompt:", "", "```", s["prompt"], "```", "", "Output:", "", "```", s["output"], "```", ""]
    with open(out / "model_card.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
