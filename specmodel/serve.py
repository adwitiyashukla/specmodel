import asyncio
import json
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from specmodel.export import load_export
from specmodel.model import generate_stream
from specmodel.tasks import get_task
from specmodel.tokenizer import chat_prompt_ids
from specmodel.train import pick_dtype


class Request:
    def __init__(self, ids, max_new_tokens, temperature, loop):
        self.ids = ids
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.loop = loop
        self.queue = asyncio.Queue()
        self.sent = ""
        self.finish_reason = None

    def push(self, item):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, item)


class Engine:
    def __init__(self, export_dir, device):
        self.model, self.tok, self.spec = load_export(export_dir, device)
        self.task = get_task(self.spec.client.task)
        self.name = self.spec.client.name
        self.device = device
        self.dtype = pick_dtype(self.spec.pretrain.dtype, device)
        self.pending = deque()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def prompt_ids(self, user):
        max_prompt = self.spec.model.seq_len - self.spec.generation.max_new_tokens
        return chat_prompt_ids(self.tok, self.spec.serve.system_prompt, user, max_prompt)

    def submit(self, request):
        with self.lock:
            self.pending.append(request)
        self.wake.set()

    def _loop(self):
        while True:
            self.wake.wait()
            time.sleep(self.spec.serve.batch_wait_ms / 1000.0)
            with self.lock:
                batch = [self.pending.popleft() for _ in range(min(len(self.pending), self.spec.serve.max_batch))]
                if not self.pending:
                    self.wake.clear()
            if batch:
                try:
                    self._run(batch)
                except Exception as exc:
                    for req in batch:
                        req.push(("error", str(exc)))

    def _run(self, batch):
        gen = self.spec.generation
        tok = self.tok
        outputs = [[] for _ in batch]
        finished = [False] * len(batch)
        stops = list(self.spec.serve.stop)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            stream = generate_stream(
                self.model,
                [r.ids for r in batch],
                tok.eos_id,
                tok.pad_id,
                [r.max_new_tokens for r in batch],
                [r.temperature for r in batch],
                gen.top_k,
                gen.top_p,
                gen.repetition_penalty,
            )
            for nxt, done in stream:
                tokens = nxt.tolist()
                flags = done.tolist()
                for i, req in enumerate(batch):
                    if finished[i]:
                        continue
                    if tokens[i] == tok.eos_id:
                        finished[i] = True
                        req.push(("done", "stop"))
                        continue
                    outputs[i].append(tokens[i])
                    text = tok.decode(outputs[i])
                    if text.endswith("\ufffd"):
                        continue
                    reason = None
                    for stop in stops:
                        cut = text.find(stop, max(0, len(req.sent) - len(stop)))
                        if cut != -1:
                            text = text[: cut + len(stop)]
                            reason = "stop"
                            break
                    if reason is None and self.task.violation(text, self.spec):
                        reason = "content_filter"
                    delta = text[len(req.sent) :]
                    req.sent = text
                    if delta:
                        req.push(("delta", delta))
                    if reason is None and flags[i]:
                        reason = "length"
                    if reason is not None:
                        finished[i] = True
                        req.push(("done", reason))
        for i, req in enumerate(batch):
            if not finished[i]:
                req.push(("done", "length"))


def _chunk(request_id, model, created, delta, finish_reason):
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }


def create_app(engines):
    app = FastAPI(title="specmodel")
    page = Path(__file__).with_name("index.html").read_text(encoding="utf-8")

    def pick(name):
        if name in engines:
            return engines[name]
        if name is None and len(engines) == 1:
            return next(iter(engines.values()))
        raise HTTPException(status_code=404, detail=f"unknown model {name}")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return page

    @app.get("/health")
    async def health():
        return {"status": "ok", "models": list(engines)}

    @app.get("/v1/models")
    async def models():
        data = [
            {
                "id": name,
                "object": "model",
                "owned_by": "specmodel",
                "description": e.spec.client.description,
                "example": e.spec.serve.example,
            }
            for name, e in engines.items()
        ]
        return {"object": "list", "data": data}

    async def run(engine, user, body):
        gen = engine.spec.generation
        max_new = int(body.get("max_tokens") or gen.max_new_tokens)
        max_new = max(1, min(max_new, gen.max_new_tokens))
        temperature = float(body.get("temperature", gen.temperature))
        req = Request(engine.prompt_ids(user), max_new, temperature, asyncio.get_running_loop())
        engine.submit(req)
        return req

    async def events(req):
        while True:
            kind, value = await req.queue.get()
            if kind == "error":
                raise HTTPException(status_code=500, detail=value)
            yield kind, value
            if kind == "done":
                return

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        engine = pick(body.get("model"))
        messages = body.get("messages") or []
        users = [m.get("content", "") for m in messages if m.get("role") == "user"]
        if not users:
            raise HTTPException(status_code=400, detail="messages must contain a user message")
        return await complete(engine, users[-1], body, chat_format=True)

    @app.post("/v1/completions")
    async def completions(body: dict):
        engine = pick(body.get("model"))
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise HTTPException(status_code=400, detail="prompt must be a string")
        return await complete(engine, prompt, body, chat_format=False)

    async def complete(engine, user, body, chat_format):
        req = await run(engine, user, body)
        request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        prompt_tokens = len(req.ids)
        if body.get("stream"):

            async def stream():
                first = {"role": "assistant", "content": ""} if chat_format else ""
                yield (
                    "data: "
                    + json.dumps(_stream_obj(request_id, engine.name, created, first, None, chat_format))
                    + "\n\n"
                )
                async for kind, value in events(req):
                    if kind == "delta":
                        payload = {"content": value} if chat_format else value
                        yield (
                            "data: "
                            + json.dumps(_stream_obj(request_id, engine.name, created, payload, None, chat_format))
                            + "\n\n"
                        )
                    else:
                        payload = {} if chat_format else ""
                        yield (
                            "data: "
                            + json.dumps(_stream_obj(request_id, engine.name, created, payload, value, chat_format))
                            + "\n\n"
                        )
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")
        finish = "stop"
        async for kind, value in events(req):
            if kind == "done":
                finish = value
        content = engine.task.finalize(req.sent)
        completion_tokens = len(engine.tok.encode_plain(content))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if chat_format:
            choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": engine.name,
                "choices": [choice],
                "usage": usage,
            }
        choice = {"index": 0, "text": content, "finish_reason": finish}
        return {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": engine.name,
            "choices": [choice],
            "usage": usage,
        }

    return app


def _stream_obj(request_id, model, created, payload, finish_reason, chat_format):
    if chat_format:
        return _chunk(request_id, model, created, payload, finish_reason)
    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": payload, "finish_reason": finish_reason}],
    }


def build_engines(export_dirs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engines = {}
    for path in export_dirs:
        engine = Engine(path, device)
        engines[engine.name] = engine
        print(
            f"loaded {engine.name} from {path} ({engine.model.num_params() / 1e6:.1f}M params, {device.type})",
            flush=True,
        )
    return engines


def serve(export_dirs, host, port):
    import uvicorn

    app = create_app(build_engines(export_dirs))
    uvicorn.run(app, host=host, port=port, log_level="info")
