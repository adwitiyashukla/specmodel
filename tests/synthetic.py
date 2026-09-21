import io
import json
import random
import tarfile
import tomllib
from pathlib import Path

NAMES = ["Lily", "Ben", "Mia", "Tom", "Sara", "Max", "Anna", "Leo", "Zoe", "Sam"]
NOUNS = (
    "cat dog ball tree park cookie rocket boat hat frog bird cake kite star moon river garden box flower "
    "truck bear duck apple book sun"
).split()
VERBS = ["ran", "jumped", "laughed", "played", "looked", "smiled", "walked", "found", "hugged", "shared", "climbed"]
ADJS = ["big", "small", "happy", "brave", "shiny", "soft", "red", "blue", "quiet", "silly", "warm", "tiny"]
WORDS = (
    "brave cookie rocket shiny quiet garden kite frog share climb giggle whisper puddle blanket lantern pebble "
    "wagon acorn ribbon bucket"
).split()
FEATURES = ["Dialogue", "Twist", "MoralValue", "BadEnding", "Foreshadowing", "Conflict"]


def _story(rng, words, features):
    name = rng.choice(NAMES)
    friend = rng.choice([n for n in NAMES if n != name])
    lines = [f"{name} was a {rng.choice(ADJS)} {rng.choice(['girl', 'boy'])} who loved the {rng.choice(NOUNS)}."]
    for w in words:
        lines.append(
            f"One day {name} saw a {rng.choice(ADJS)} {w} near the {rng.choice(NOUNS)} and {rng.choice(VERBS)}."
        )
    if "Dialogue" in features:
        lines.append(f'"Look at the {rng.choice(NOUNS)}!" said {name}. "Can we play?" asked {friend}.')
    for _ in range(rng.randint(2, 7)):
        lines.append(
            f"{rng.choice([name, friend])} {rng.choice(VERBS)} to the {rng.choice(ADJS)} {rng.choice(NOUNS)} "
            f"with the {rng.choice(NOUNS)}."
        )
    if "MoralValue" in features:
        lines.append(f"{name} learned that it is good to {rng.choice(['share', 'help', 'listen', 'be kind'])}.")
    lines.append(
        f"Then {name} and {friend} went home and had a {rng.choice(ADJS)} {rng.choice(['nap', 'dinner', 'dream'])}."
    )
    return " ".join(lines)


def story_rows(n, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        words = rng.sample(WORDS, 3)
        features = rng.sample(FEATURES, rng.randint(0, 2))
        story = _story(rng, words, features)
        rows.append(
            {
                "story": story,
                "instruction": {"prompt:": "Write a short story", "words": words, "features": features},
                "summary": f"{words[0]} and {words[1]}",
                "source": "GPT-4",
            }
        )
    for i in range(n // 20):
        rows.append(dict(rows[rng.randrange(n)]))
    for i in range(n // 20):
        row = dict(rows[rng.randrange(n)])
        row["story"] = row["story"] + " The end."
        rows.append(row)
    rng.shuffle(rows)
    return rows


def write_story_tar(path, n, seed=0, shards=3):
    rows = story_rows(n, seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for s in range(shards):
            data = json.dumps(rows[s::shards]).encode("utf-8")
            info = tarfile.TarInfo(name=f"data{s:02d}.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return len(rows)


TABLES = {
    "orders": [("id", "INT"), ("customer", "TEXT"), ("amount", "DECIMAL(10,2)"), ("region", "TEXT")],
    "employees": [("id", "INT"), ("name", "TEXT"), ("salary", "INT"), ("department", "TEXT")],
    "products": [("id", "INT"), ("title", "TEXT"), ("price", "DECIMAL(8,2)"), ("category", "TEXT")],
    "visits": [("id", "INT"), ("patient", "TEXT"), ("cost", "INT"), ("clinic", "TEXT")],
}
TEXT_VALUES = ["north", "south", "east", "west", "alpha", "beta", "gamma", "delta"]


def _rows(rng, table):
    cols = TABLES[table]
    out = []
    for i in range(rng.randint(3, 6)):
        values = []
        for name, kind in cols:
            if name == "id":
                values.append(str(i + 1))
            elif kind == "TEXT":
                values.append("'" + rng.choice(TEXT_VALUES) + "'")
            elif kind.startswith("DECIMAL"):
                values.append(f"{rng.randint(1, 500)}.{rng.randint(0, 99):02d}")
            else:
                values.append(str(rng.randint(10, 900)))
        out.append("(" + ", ".join(values) + ")")
    return out


def sql_rows(n, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        table = rng.choice(list(TABLES))
        cols = TABLES[table]
        text_col = [c for c, k in cols if k == "TEXT" and c != "id"][-1]
        num_col = next(c for c, k in cols if k != "TEXT" and c != "id")
        context = (
            f"CREATE TABLE {table} (" + ", ".join(f"{c} {k}" for c, k in cols) + ");\n"
            f"INSERT INTO {table} VALUES " + ", ".join(_rows(rng, table)) + ";"
        )
        kind = rng.randint(0, 6)
        value = rng.choice(TEXT_VALUES)
        threshold = rng.randint(50, 400)
        if kind == 0:
            q, sql = f"How many rows are in {table}?", f"SELECT COUNT(*) FROM {table};"
        elif kind == 1:
            q, sql = (
                f"What is the total {num_col} per {text_col} in {table}?",
                f"SELECT {text_col}, SUM({num_col}) FROM {table} GROUP BY {text_col};",
            )
        elif kind == 2:
            q, sql = (
                f"List the {text_col} values in {table} where {num_col} is above {threshold}.",
                f"SELECT {text_col} FROM {table} WHERE {num_col} > {threshold};",
            )
        elif kind == 3:
            q, sql = f"What is the average {num_col} in {table}?", f"SELECT AVG({num_col}) FROM {table};"
        elif kind == 4:
            q, sql = (
                f"Delete every row of {table} whose {text_col} is {value}.",
                f"DELETE FROM {table} WHERE {text_col} = '{value}';",
            )
        elif kind == 5:
            q, sql = (
                f"Find rows of {table} whose {text_col} matches {value} ignoring case.",
                f"SELECT * FROM {table} WHERE {text_col} ILIKE '{value}';",
            )
        else:
            q, sql = f"What is the highest {num_col} in {table}?", f"SELECT MAX({num_col}) FROM {table};"
        rows.append(
            {
                "id": i,
                "domain": table,
                "domain_description": f"{table} data",
                "sql_complexity": "basic",
                "sql_complexity_description": "basic",
                "sql_task_type": "analytics",
                "sql_task_type_description": "analytics",
                "sql_prompt": q,
                "sql_context": context,
                "sql": sql,
                "sql_explanation": f"This query answers: {q}",
            }
        )
    return rows


def write_sql_parquet(path, n, seed=0):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = sql_rows(n, seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return len(rows)


def small_spec(base, out_path, sources, overrides):
    with open(base, "rb") as f:
        raw = tomllib.load(f)
    raw["data"]["sources"] = sources
    for section, values in overrides.items():
        raw[section].update(values)
    lines = []
    for section, values in raw.items():
        if section == "data":
            plain = {k: v for k, v in values.items() if k != "sources"}
            lines.append(_table("data", plain))
            for source in values["sources"]:
                lines.append(_table("[data.sources]", source))
        elif section == "eval":
            plain = {k: v for k, v in values.items() if k != "gates"}
            lines.append(_table("eval", plain))
            for gate in values["gates"]:
                lines.append(_table("[eval.gates]", gate))
        else:
            lines.append(_table(section, values))
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    return json.dumps(v)


def _table(name, values):
    out = [f"[{name}]"]
    for k, v in values.items():
        out.append(f"{k} = {_value(v)}")
    return "\n".join(out) + "\n"
