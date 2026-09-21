import tomllib
from pathlib import Path

import pytest

from specmodel.spec import SpecError, load_spec, validate

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", ["storyteller", "sqlbot"])
def test_client_specs_load(name):
    spec = load_spec(ROOT / "clients" / name / "spec.toml")
    assert spec.client.name == name
    assert spec.model.dim % spec.model.n_heads == 0
    assert spec.eval.gates
    assert isinstance(spec.serve.stop, list)


def _raw(name="sqlbot"):
    with open(ROOT / "clients" / name / "spec.toml", "rb") as f:
        return tomllib.load(f)


def test_rejects_bad_head_split():
    raw = _raw()
    raw["model"]["n_heads"] = 5
    with pytest.raises(SpecError, match="divisible"):
        validate(raw)


def test_rejects_missing_field():
    raw = _raw()
    del raw["pretrain"]["lr"]
    with pytest.raises(SpecError, match="pretrain.lr"):
        validate(raw)


def test_rejects_bad_choice_and_range():
    raw = _raw()
    raw["data"]["dedup"] = "fuzzy"
    with pytest.raises(SpecError, match="dedup"):
        validate(raw)
    raw = _raw()
    raw["generation"]["top_p"] = 1.5
    with pytest.raises(SpecError, match="top_p"):
        validate(raw)


def test_rejects_gate_without_bound():
    raw = _raw()
    raw["eval"]["gates"] = [{"metric": "pass_rate"}]
    with pytest.raises(SpecError, match="min or max"):
        validate(raw)


def test_rejects_bool_as_int():
    raw = _raw()
    raw["model"]["n_layers"] = True
    with pytest.raises(SpecError, match="integer"):
        validate(raw)
