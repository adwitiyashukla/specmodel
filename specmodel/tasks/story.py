import re

WORD = re.compile(r"[A-Za-z']+")


def _clean(text):
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()


def record(row, spec):
    story = row.get("story")
    if not isinstance(story, str):
        return None
    story = _clean(story)
    instruction = row.get("instruction") or {}
    words = [w.strip().lower() for w in instruction.get("words") or [] if isinstance(w, str) and w.strip()]
    features = [f.strip() for f in instruction.get("features") or [] if isinstance(f, str) and f.strip()]
    return {
        "text": story,
        "prompt": build_prompt(words, features),
        "response": story,
        "meta": {"words": words, "features": features},
    }


def build_prompt(words, features):
    parts = ["Write a short story for a child."]
    if words:
        parts.append("Use these words: " + ", ".join(words) + ".")
    if features:
        parts.append("Include: " + ", ".join(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", f).lower() for f in features) + ".")
    return " ".join(parts)


def _has_word(word, text):
    return re.search(r"\b" + re.escape(word) + r"(s|es|ed|d|ing)?\b", text, flags=re.IGNORECASE) is not None


def score(rec, output, spec):
    task = spec.task
    words = rec["meta"].get("words", [])
    features = rec["meta"].get("features", [])
    n_words = len(WORD.findall(output))
    hits = [_has_word(w, output) for w in words]
    coverage = sum(hits) / len(hits) if hits else 1.0
    all_words = all(hits) if hits else True
    dialogue_ok = ('"' in output) if "Dialogue" in features else True
    banned = any(_has_word(w, output) for w in task.banned_words)
    length_ok = task.min_words <= n_words <= task.max_words
    ended = output.rstrip().endswith((".", "!", "?", '"'))
    passed = all_words and dialogue_ok and not banned and length_ok and ended
    return {
        "word_coverage": coverage,
        "all_words": float(all_words),
        "dialogue": float(dialogue_ok),
        "banned": float(banned),
        "length_ok": float(length_ok),
        "ended": float(ended),
        "passed": float(passed),
        "n_words": float(n_words),
    }


def summarize(rows):
    n = max(len(rows), 1)
    return {
        "word_coverage": sum(r["word_coverage"] for r in rows) / n,
        "all_words_rate": sum(r["all_words"] for r in rows) / n,
        "dialogue_rate": sum(r["dialogue"] for r in rows) / n,
        "banned_rate": sum(r["banned"] for r in rows) / n,
        "length_rate": sum(r["length_ok"] for r in rows) / n,
        "ended_rate": sum(r["ended"] for r in rows) / n,
        "pass_rate": sum(r["passed"] for r in rows) / n,
        "mean_words": sum(r["n_words"] for r in rows) / n,
    }


def passed(scores):
    return scores["passed"] >= 1.0


def finalize(output):
    return output.strip()


def violation(text, spec):
    return any(_has_word(w, text) for w in spec.task.banned_words)
