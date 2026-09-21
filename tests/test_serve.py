import asyncio
import json

import httpx
import pytest

from specmodel.serve import build_engines, create_app

QUESTION = (
    "Schema:\nCREATE TABLE orders (id INT, customer TEXT, amount DECIMAL(10,2));\n"
    "INSERT INTO orders VALUES (1, 'ana', 12.5);\n\nQuestion: How many rows are in orders?"
)


@pytest.fixture(scope="module")
def app(sql_run, story_run):
    return create_app(build_engines([sql_run / "export", story_run / "export"]))


def run(app, coro_fn):
    async def inner():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as client:
            return await coro_fn(client)

    return asyncio.run(inner())


def test_models_health_and_page(app):
    async def go(client):
        models = (await client.get("/v1/models")).json()
        health = (await client.get("/health")).json()
        page = await client.get("/")
        return models, health, page

    models, health, page = run(app, go)
    assert [m["id"] for m in models["data"]] == ["sqlbot", "storyteller"]
    assert models["data"][0]["example"].startswith("Schema:")
    assert health == {"status": "ok", "models": ["sqlbot", "storyteller"]}
    assert page.status_code == 200 and '<select id="model">' in page.text


def test_chat_completion_shape_and_stop(app):
    async def go(client):
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "sqlbot", "max_tokens": 16, "messages": [{"role": "user", "content": QUESTION}]},
        )
        return r.status_code, r.json()

    status, body = run(app, go)
    assert status == 200
    assert body["object"] == "chat.completion" and body["model"] == "sqlbot"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant" and isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in ("stop", "length")
    assert choice["message"]["content"].count(";") <= 1
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]


def test_streaming_matches_openai_chunks(app):
    async def go(client):
        chunks = []
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "storyteller",
                "stream": True,
                "max_tokens": 12,
                "messages": [{"role": "user", "content": "Write a short story for a child. Use these words: frog."}],
            },
        ) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line[6:])
        return chunks

    chunks = run(app, go)
    assert chunks[-1] == "[DONE]"
    parsed = [json.loads(c) for c in chunks[:-1]]
    assert parsed[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert all(p["object"] == "chat.completion.chunk" for p in parsed)
    assert parsed[-1]["choices"][0]["finish_reason"] in ("stop", "length", "content_filter")
    text = "".join(p["choices"][0]["delta"].get("content", "") for p in parsed)
    assert isinstance(text, str)


def test_completions_endpoint_and_batching(app):
    async def go(client):
        single = await client.post(
            "/v1/completions", json={"model": "storyteller", "prompt": "Write a story.", "max_tokens": 8}
        )
        many = await asyncio.gather(
            *[
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "sqlbot",
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": QUESTION + str(i)}],
                    },
                )
                for i in range(6)
            ]
        )
        return single.json(), [r.json() for r in many]

    single, many = run(app, go)
    assert single["object"] == "text_completion" and isinstance(single["choices"][0]["text"], str)
    assert len(many) == 6 and all(r["choices"][0]["message"]["content"] is not None for r in many)


def test_bad_requests(app):
    async def go(client):
        a = await client.post(
            "/v1/chat/completions", json={"model": "nope", "messages": [{"role": "user", "content": "x"}]}
        )
        b = await client.post("/v1/chat/completions", json={"model": "sqlbot", "messages": []})
        c = await client.post("/v1/completions", json={"model": "sqlbot"})
        return a.status_code, b.status_code, c.status_code

    assert run(app, go) == (404, 400, 400)
