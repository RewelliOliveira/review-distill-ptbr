import pytest
from google.genai import types
from pydantic import ValidationError

from review_distill.schema import (
    ASPECT_NAMES,
    LabeledReview,
    MainIssue,
    ReviewDiagnosis,
    Sentiment,
    gemini_batch_response_schema,
)

VALID = {
    "aspects": {
        "delivery_time": "negative",
        "product_quality": "positive",
        "as_described": "not_mentioned",
        "order_accuracy": "not_mentioned",
        "packaging": "not_mentioned",
        "customer_service": "negative",
    },
    "main_issue": "late_delivery",
    "needs_contact": True,
}


def test_valid_diagnosis_parses() -> None:
    d = ReviewDiagnosis.model_validate(VALID)
    assert d.aspects.delivery_time is Sentiment.NEGATIVE
    assert d.main_issue is MainIssue.LATE_DELIVERY
    assert d.needs_contact is True


def test_aspect_names_match_claude_md() -> None:
    assert ASPECT_NAMES == (
        "delivery_time",
        "product_quality",
        "as_described",
        "order_accuracy",
        "packaging",
        "customer_service",
    )


def test_main_issue_values() -> None:
    assert {m.value for m in MainIssue} == {
        "none",
        "late_delivery",
        "not_delivered",
        "wrong_item",
        "missing_item",
        "damaged",
        "poor_quality",
        "not_as_described",
        "bad_service",
        "other",
    }


def test_roundtrip_json() -> None:
    d = ReviewDiagnosis.model_validate(VALID)
    assert ReviewDiagnosis.model_validate_json(d.model_dump_json()) == d


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x["aspects"].update(delivery_time="neutral"),
        lambda x: x["aspects"].pop("packaging"),
        lambda x: x["aspects"].update(price="negative"),
        lambda x: x.update(main_issue="slow"),
        lambda x: x.pop("needs_contact"),
        lambda x: x.update(needs_contact="maybe"),
        lambda x: x.update(extra_field=1),
    ],
)
def test_invalid_diagnosis_rejected(mutate) -> None:
    data = {**VALID, "aspects": dict(VALID["aspects"])}
    mutate(data)
    with pytest.raises(ValidationError):
        ReviewDiagnosis.model_validate(data)


def test_labeled_review_requires_id() -> None:
    with pytest.raises(ValidationError):
        LabeledReview.model_validate(VALID)
    with pytest.raises(ValidationError):
        LabeledReview.model_validate({**VALID, "review_id": ""})
    item = LabeledReview.model_validate({**VALID, "review_id": "abc"})
    assert item.diagnosis() == ReviewDiagnosis.model_validate(VALID)


def test_gemini_schema_is_accepted_by_sdk() -> None:
    schema = types.Schema.model_validate(gemini_batch_response_schema())
    item = schema.items
    assert schema.type == types.Type.ARRAY
    assert item.required == ["review_id", "aspects", "main_issue", "needs_contact"]
    assert item.properties["main_issue"].enum == [m.value for m in MainIssue]
    assert list(item.properties["aspects"].properties) == list(ASPECT_NAMES)
