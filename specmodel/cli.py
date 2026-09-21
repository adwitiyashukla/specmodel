import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from specmodel.spec import load_spec
from specmodel.tasks import get_task

STAGES = ("data", "tokenizer", "tokenize", "pretrain", "sft", "mine", "dpo", "export")


def run_dir_for(args, spec):
    return Path(args.runs) / spec.client.name


def done_marker(run_dir, stage):
    return {
        "data": run_dir / "data" / "stats.json",
        "tokenizer": run_dir / "tokenizer" / "tokenizer.json",
        "tokenize": run_dir / "shards" / "meta.json",
        "pretrain": run_dir / "pretrain" / "summary.json",
        "sft": run_dir / "sft" / "summary.json",
        "mine": run_dir / "dpo" / "pairs.jsonl",
        "dpo": run_dir / "dpo" / "summary.json",
        "export": run_dir / "export" / "config.json",
    }[stage]


def run_stage(stage, spec, run_dir, args):
    task = get_task(spec.client.task)
    if stage == "data":
        from specmodel.data import build_records

        build_records(spec, task, run_dir, args.cache)
    elif stage == "tokenizer":
        from specmodel.data import train_tokenizer

        train_tokenizer(spec, run_dir)
    elif stage == "tokenize":
        from specmodel.data import write_shards

        write_shards(spec, run_dir)
    elif stage == "pretrain":
        run_pretrain(spec, run_dir, args)
    elif stage == "sft":
        from specmodel.posttrain import sft

        sft(spec, run_dir)
    elif stage == "mine":
        from specmodel.posttrain import mine_pairs

        mine_pairs(spec, run_dir, task)
    elif stage == "dpo":
        from specmodel.posttrain import dpo

        dpo(spec, run_dir)
    elif stage == "export":
        from specmodel.export import export

        export(spec, run_dir)
    else:
        raise ValueError(stage)


def run_pretrain(spec, run_dir, args):
    import torch

    gpus = torch.cuda.device_count()
    if gpus > 1 and "WORLD_SIZE" not in os.environ:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={gpus}",
            "-m",
            "specmodel",
            "--runs",
            args.runs,
            "--cache",
            args.cache,
            "pretrain",
            "--spec",
            args.spec,
        ]
        print(f"pretrain: launching {gpus} processes", flush=True)
        subprocess.run(command, check=True)
        return
    from specmodel.train import pretrain

    pretrain(spec, run_dir)


def cmd_stage(args):
    spec = load_spec(args.spec)
    run_stage(args.command, spec, run_dir_for(args, spec), args)


def cmd_eval(args):
    from specmodel.evaluate import evaluate

    spec = load_spec(args.spec)
    run_dir = run_dir_for(args, spec)
    task = get_task(spec.client.task)
    failed = False
    for stage in args.stage:
        report = evaluate(spec, run_dir, stage, task)
        failed = failed or not report["gates_passed"]
    if failed and args.strict:
        sys.exit(1)


def cmd_build(args):
    from specmodel.evaluate import evaluate

    spec = load_spec(args.spec)
    run_dir = run_dir_for(args, spec)
    task = get_task(spec.client.task)
    for stage in STAGES:
        if done_marker(run_dir, stage).exists():
            print(f"{stage}: done", flush=True)
        else:
            print(f"{stage}: running", flush=True)
            run_stage(stage, spec, run_dir, args)
        if stage in ("pretrain", "sft", "dpo", "export") and not (run_dir / "eval" / f"{stage}.json").exists():
            evaluate(spec, run_dir, stage, task)
    with open(run_dir / "eval" / "export.json", encoding="utf-8") as f:
        final = json.load(f)
    for gate in final["gates"]:
        print(f"gate {gate['metric']}: {gate['value']:.3f} {'ok' if gate['ok'] else 'FAILED'}", flush=True)
    if not final["gates_passed"]:
        print("build failed: gates not met", flush=True)
        sys.exit(1)
    print("build passed", flush=True)


def cmd_inspect(args):
    from specmodel.data import iter_rows, source_path

    spec = load_spec(args.spec)
    task = get_task(spec.client.task)
    for source in spec.data.sources:
        path = source_path(source, args.cache)
        print(f"source {source['name']} at {path}")
        for i, row in enumerate(iter_rows(source["format"], path)):
            if i >= args.n:
                break
            rec = task.record(row, spec)
            print(json.dumps({"row_keys": sorted(row.keys()), "record": rec}, ensure_ascii=False, indent=2)[:3000])


def cmd_serve(args):
    from specmodel.serve import serve

    serve(args.export, args.host, args.port)


def cmd_chat(args):
    import torch

    from specmodel.export import load_export
    from specmodel.model import generate
    from specmodel.tokenizer import chat_prompt_ids

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tok, spec = load_export(args.export, device)
    task = get_task(spec.client.task)
    gen = spec.generation
    max_prompt = spec.model.seq_len - gen.max_new_tokens
    print(f"{spec.client.name}: {spec.client.description}")
    print("type a prompt, empty line to quit")
    while True:
        try:
            user = input("> ").strip()
        except EOFError:
            break
        if not user:
            break
        ids = chat_prompt_ids(tok, spec.serve.system_prompt, user, max_prompt)
        out = generate(
            model,
            [ids],
            tok.eos_id,
            tok.pad_id,
            gen.max_new_tokens,
            gen.temperature,
            gen.top_k,
            gen.top_p,
            gen.repetition_penalty,
        )[0]
        print(task.finalize(tok.decode(out)))
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="specmodel")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--cache", default="data")
    sub = parser.add_subparsers(dest="command", required=True)
    for stage in STAGES:
        p = sub.add_parser(stage)
        p.add_argument("--spec", required=True)
        p.set_defaults(func=cmd_stage)
    p = sub.add_parser("eval")
    p.add_argument("--spec", required=True)
    p.add_argument("--stage", nargs="+", default=["export"])
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser("build")
    p.add_argument("--spec", required=True)
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("inspect")
    p.add_argument("--spec", required=True)
    p.add_argument("--n", type=int, default=2)
    p.set_defaults(func=cmd_inspect)
    p = sub.add_parser("serve")
    p.add_argument("--export", nargs="+", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    p = sub.add_parser("chat")
    p.add_argument("--export", required=True)
    p.set_defaults(func=cmd_chat)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
