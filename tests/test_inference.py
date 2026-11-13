import json

import pytest

from review_distill.inference import (
    diagnosis_to_json,
    parse_diagnosis,
    prediction_record,
    training_example,
)
from review_distill.schema import ASPECT_NAMES, ReviewDiagnosis

DIAG = ReviewDiagnosis.model_validate(
    {
        "aspects": dict.fromkeys(ASPECT_NAMES, "not_mentioned") | {"delivery_time": "negative"},
        "main_issue": "late_delivery",
        "needs_contact": False,
    }
)


def test_training_example_roundtrips_through_parser():
    ex = training_example("Chegou atrasado", DIAG)
    assert [m["role"] for m in ex["prompt"]] == ["system", "user"]
    assert ex["prompt"][1]["content"] == "Chegou atrasado"
    (answer,) = ex["completion"]
    assert answer["role"] == "assistant"
    assert parse_diagnosis(answer["content"]) == DIAG


def test_diagnosis_json_is_single_line_and_keeps_accents():
    out = diagnosis_to_json(DIAG)
    assert "\n" not in out
    assert json.loads(out)["main_issue"] == "late_delivery"


@pytest.mark.parametrize(
    "wrap",
    [
        "{}",
        "  {}\n",
        "```json\n{}\n```",
        "Resposta: {} fim",
    ],
)
def test_parse_tolerates_wrapping(wrap):
    assert parse_diagnosis(wrap.replace("{}", diagnosis_to_json(DIAG))) == DIAG


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "sem json",
        '{"aspects": {}',  # truncated
        diagnosis_to_json(DIAG).replace("late_delivery", "atraso"),  # bad enum
        diagnosis_to_json(DIAG)[:-1] + ', "extra": 1}',  # extra key
    ],
)
def test_parse_rejects_invalid(raw):
    assert parse_diagnosis(raw) is None


def test_prediction_record():
    assert prediction_record("a", "{}", 0.0123456) == {
        "review_id": "a",
        "raw": "{}",
        "latency_s": 0.01235,
    }
