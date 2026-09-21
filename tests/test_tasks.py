from specmodel.spec import load_spec
from specmodel.tasks import sql, story

CONTEXT = (
    "CREATE TABLE orders (id INT, customer TEXT, amount DECIMAL(10,2));\n"
    "INSERT INTO orders VALUES (1, 'ana', 120.5), (2, 'ben', 80.0), (3, 'ana', 45.25);"
)


def story_record(words, features):
    return {"meta": {"words": words, "features": features}}


def test_story_scoring(story_spec):
    spec = load_spec(story_spec)
    rec = story_record(["frog", "kite"], ["Dialogue"])
    good = ("The frog and the kite were friends. " * 8) + '"Hello," said the frog. ' + ("They played all day. " * 10)
    s = story.score(rec, good, spec)
    assert s["passed"] == 1.0 and s["word_coverage"] == 1.0 and s["banned"] == 0.0 and s["dialogue"] == 1.0
    cut = story.score(rec, good + " Then the frog", spec)
    assert cut["ended"] == 0.0 and cut["passed"] == 0.0
    missing = story.score(rec, "The frogs flew kites happily. " * 20, spec)
    assert missing["word_coverage"] == 1.0
    no_quotes = story.score(rec, "The frog liked the kite. " * 20, spec)
    assert no_quotes["dialogue"] == 0.0 and no_quotes["passed"] == 0.0
    banned = story.score(rec, good + " The gun was loud.", spec)
    assert banned["banned"] == 1.0 and banned["passed"] == 0.0
    short = story.score(rec, '"Frog kite," she said.', spec)
    assert short["length_ok"] == 0.0
    summary = story.summarize([s, missing, no_quotes, banned])
    assert summary["pass_rate"] == 0.25 and summary["banned_rate"] == 0.25 and summary["ended_rate"] == 1.0
    assert story.violation("a gun", spec) and not story.violation("a sun", spec)
    assert (
        story.build_prompt(["a"], ["BadEnding"])
        == "Write a short story for a child. Use these words: a. Include: bad ending."
    )


def test_story_record_parses_tinystories_rows(story_spec):
    spec = load_spec(story_spec)
    row = {"story": "Once upon a time " * 20, "instruction": {"words": [" Frog", "kite"], "features": ["Dialogue"]}}
    rec = story.record(row, spec)
    assert rec["meta"]["words"] == ["frog", "kite"]
    assert rec["prompt"].startswith("Write a short story")
    assert story.record({"instruction": {}}, spec) is None


def test_sql_execution_scoring(sql_spec):
    spec = load_spec(sql_spec)
    rec = {"meta": {"context": CONTEXT, "sql": "SELECT customer, SUM(amount) FROM orders GROUP BY customer;"}}
    same = sql.score(rec, "select customer, sum(amount) from orders group by customer", spec)
    assert same["exec"] == 1.0 and same["valid"] == 1.0 and same["gold_runs"] == 1.0 and same["exact"] == 1.0
    reordered = sql.score(
        rec, "SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer ORDER BY 1 DESC;", spec
    )
    assert reordered["exec"] == 1.0 and reordered["exact"] == 0.0
    wrong = sql.score(rec, "SELECT customer, COUNT(*) FROM orders GROUP BY customer;", spec)
    assert wrong["exec"] == 0.0 and wrong["valid"] == 1.0
    broken = sql.score(rec, "SELECT customer FROM nowhere;", spec)
    assert broken["valid"] == 0.0 and broken["passed"] == 0.0
    fenced = sql.score(
        rec, "```sql\nSELECT customer, SUM(amount) FROM orders GROUP BY customer;\n``` extra words", spec
    )
    assert fenced["exec"] == 1.0


def test_sql_writes_compare_table_state(sql_spec):
    spec = load_spec(sql_spec)
    rec = {"meta": {"context": CONTEXT, "sql": "DELETE FROM orders WHERE customer = 'ana';"}}
    assert sql.score(rec, "DELETE FROM orders WHERE customer='ana';", spec)["exec"] == 1.0
    assert sql.score(rec, "DELETE FROM orders WHERE customer = 'ben';", spec)["exec"] == 0.0


def test_sql_unrunnable_gold_is_excluded_from_execution_accuracy(sql_spec):
    spec = load_spec(sql_spec)
    rec = {"meta": {"context": CONTEXT, "sql": "SELECT * FROM orders WHERE customer ILIKE 'ana';"}}
    copied = sql.score(rec, "SELECT * FROM orders WHERE customer ILIKE 'ana';", spec)
    assert copied["gold_runs"] == 0.0 and copied["valid"] == 0.0 and copied["exact"] == 1.0 and copied["passed"] == 1.0
    summary = sql.summarize(
        [
            copied,
            sql.score(
                {"meta": {"context": CONTEXT, "sql": "SELECT COUNT(*) FROM orders;"}},
                "SELECT COUNT(*) FROM orders;",
                spec,
            ),
        ]
    )
    assert summary["execution_accuracy"] == 1.0 and summary["gold_executable_rate"] == 0.5


def test_sql_timeout_and_finalize():
    slow = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r;"
    assert sql.run_query(CONTEXT, slow, 200, 10) is None
    assert sql.finalize("SELECT 1; SELECT 2;") == "SELECT 1;"
    assert sql.finalize("  select 1  ") == "select 1"
    assert sql.normalize("SELECT  a ,b FROM t ;") == "select a,b from t"
