"""Output schema for review diagnoses. This module is the source of truth.

Any change here requires re-labeling the affected data and updating the tests.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NOT_MENTIONED = "not_mentioned"


class MainIssue(StrEnum):
    NONE = "none"
    LATE_DELIVERY = "late_delivery"
    NOT_DELIVERED = "not_delivered"
    WRONG_ITEM = "wrong_item"
    MISSING_ITEM = "missing_item"
    DAMAGED = "damaged"
    POOR_QUALITY = "poor_quality"
    NOT_AS_DESCRIBED = "not_as_described"
    BAD_SERVICE = "bad_service"
    OTHER = "other"


class Aspects(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delivery_time: Sentiment = Field(description="Delivery time / delays.")
    product_quality: Sentiment = Field(description="Quality of the product itself.")
    as_described: Sentiment = Field(description="Product matches the listing.")
    order_accuracy: Sentiment = Field(description="Wrong or missing items.")
    packaging: Sentiment = Field(description="Packaging condition.")
    customer_service: Sentiment = Field(description="Seller / customer service.")


ASPECT_NAMES: tuple[str, ...] = tuple(Aspects.model_fields)


class ReviewDiagnosis(BaseModel):
    """Structured diagnosis of a single review."""

    model_config = ConfigDict(extra="forbid")

    aspects: Aspects
    main_issue: MainIssue
    needs_contact: bool


class LabeledReview(ReviewDiagnosis):
    """A diagnosis tied to its review id, as returned by the labeler in batch mode."""

    review_id: str = Field(min_length=1)

    def diagnosis(self) -> ReviewDiagnosis:
        return ReviewDiagnosis.model_validate(self.model_dump(exclude={"review_id"}))


def _enum_schema(enum: type[StrEnum]) -> dict[str, Any]:
    return {"type": "STRING", "format": "enum", "enum": [e.value for e in enum]}


def gemini_batch_response_schema() -> dict[str, Any]:
    """Gemini response schema (OpenAPI subset) for a list of LabeledReview objects.

    Written explicitly (no $refs, explicit ordering) so the API sees a flat schema;
    every returned item is still validated with pydantic.
    """
    aspects = {
        "type": "OBJECT",
        "properties": {name: _enum_schema(Sentiment) for name in ASPECT_NAMES},
        "required": list(ASPECT_NAMES),
        "property_ordering": list(ASPECT_NAMES),
    }
    item_fields = ["review_id", "aspects", "main_issue", "needs_contact"]
    item = {
        "type": "OBJECT",
        "properties": {
            "review_id": {"type": "STRING"},
            "aspects": aspects,
            "main_issue": _enum_schema(MainIssue),
            "needs_contact": {"type": "BOOLEAN"},
        },
        "required": item_fields,
        "property_ordering": item_fields,
    }
    return {"type": "ARRAY", "items": item}
