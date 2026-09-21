import re
import sqlite3
import time

FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def _clean(text):
    return text.replace("\r\n", "\n").strip()


def record(row, spec):
    question = row.get("sql_prompt")
    context = row.get("sql_context")
    sql = row.get("sql")
    if not all(isinstance(v, str) and v.strip() for v in (question, context, sql)):
        return None
    question, context, sql = _clean(question), _clean(context), _clean(sql)
    explanation = row.get("sql_explanation")
    text = context + "\n\n-- " + question + "\n" + sql
    if isinstance(explanation, str) and explanation.strip():
        text += "\n\n-- " + _clean(explanation)
    return {
        "text": text,
        "prompt": build_prompt(context, question),
        "response": sql,
        "meta": {"context": context, "sql": sql, "domain": row.get("domain") or ""},
    }


def build_prompt(context, question):
    return "Schema:\n" + context + "\n\nQuestion: " + question


def normalize(sql):
    sql = re.sub(r"\s+", " ", sql.strip().rstrip(";").strip()).lower()
    return re.sub(r"\s*([(),=<>])\s*", r"\1", sql)


def _is_read(sql):
    head = sql.lstrip().lower()
    return head.startswith(("select", "with"))


def _snapshot(conn, max_rows):
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    state = []
    for name in tables:
        rows = conn.execute(f'SELECT * FROM "{name}"').fetchmany(max_rows)
        state.append((name, sorted(tuple(map(str, r)) for r in rows)))
    return state


def run_query(context, sql, timeout_ms, max_rows):
    conn = sqlite3.connect(":memory:")
    deadline = time.monotonic() + timeout_ms / 1000.0

    def guard():
        return 1 if time.monotonic() > deadline else 0

    try:
        conn.executescript(context)
    except sqlite3.Error:
        conn.close()
        return None
    conn.set_progress_handler(guard, 1000)
    try:
        if _is_read(sql):
            rows = conn.execute(sql).fetchmany(max_rows)
            result = sorted(tuple(map(str, r)) for r in rows)
        else:
            conn.executescript(sql)
            result = _snapshot(conn, max_rows)
    except (sqlite3.Error, OverflowError, RecursionError):
        result = None
    finally:
        conn.close()
    return result


def finalize(output):
    m = FENCE.search(output)
    if m:
        output = m.group(1)
    output = output.strip()
    cut = output.find(";")
    if cut != -1:
        output = output[: cut + 1]
    return output.strip()


def score(rec, output, spec):
    task = spec.task
    context = rec["meta"]["context"]
    gold = rec["meta"]["sql"]
    pred = finalize(output)
    gold_rows = run_query(context, gold, task.timeout_ms, task.max_rows)
    pred_rows = run_query(context, pred, task.timeout_ms, task.max_rows) if pred else None
    gold_runs = gold_rows is not None
    valid = pred_rows is not None
    exact = normalize(pred) == normalize(gold) if pred else False
    execution = gold_runs and valid and pred_rows == gold_rows
    return {
        "valid": float(valid),
        "exact": float(exact),
        "gold_runs": float(gold_runs),
        "exec": float(execution),
        "passed": float(execution or (not gold_runs and exact)),
    }


def summarize(rows):
    n = max(len(rows), 1)
    runnable = sum(r["gold_runs"] for r in rows)
    return {
        "valid_rate": sum(r["valid"] for r in rows) / n,
        "exact_match": sum(r["exact"] for r in rows) / n,
        "gold_executable_rate": runnable / n,
        "execution_accuracy": sum(r["exec"] for r in rows) / max(runnable, 1),
        "pass_rate": sum(r["passed"] for r in rows) / n,
    }


def passed(scores):
    return scores["passed"] >= 1.0


def violation(text, spec):
    return False
