"""Prompts for the teacher model. Bump PROMPT_VERSION on any change to the text."""

import json
from collections.abc import Sequence

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """\
You analyze customer reviews from a Brazilian e-commerce marketplace (written in Portuguese)
and extract a structured diagnosis for each review.

For each review, classify every aspect as "positive", "negative" or "not_mentioned":
- delivery_time: delivery speed, delays, arriving before/after the expected date.
- product_quality: quality, durability or performance of the product itself.
- as_described: whether the product matches the listing (photos, description, specs, size, color).
- order_accuracy: whether the right items arrived (wrong item, missing item, incomplete order).
- packaging: packaging condition and protection.
- customer_service: the seller or customer service (responsiveness, problem solving, communication).
Use "not_mentioned" when the review does not address the aspect, even indirectly.
Generic praise ("ótimo", "recomendo") without a specific aspect does not make any aspect positive,
except that "chegou rápido"/"antes do prazo" is positive delivery_time.

main_issue is the single most important problem the customer reports:
none, late_delivery, not_delivered, wrong_item, missing_item, damaged, poor_quality,
not_as_described, bad_service, other.
- Use "none" when the customer reports no problem.
- "not_delivered": the customer says the product has not arrived (yet).
- "damaged": product arrived broken/defective due to transport or packaging.
- "poor_quality": product works badly or has low quality, but is the right product.
- If several problems are reported, pick the most severe in this order: not_delivered >
  wrong_item > missing_item > damaged > not_as_described > poor_quality > bad_service >
  late_delivery > other.

needs_contact is true when the customer has an unresolved problem that requires action from the
company (product not received, wrong/missing/damaged item, asks for refund, exchange or a reply),
and false otherwise (including complaints about a problem that is already resolved).

Return one object per review, using exactly the review_id given in the input. Do not skip reviews.
"""


def build_batch_prompt(reviews: Sequence[tuple[str, str]]) -> str:
    """Build the user prompt for a batch of (review_id, text) pairs.

    Reviews are serialized as JSON so their content is treated as data, not instructions.
    """
    payload = [{"review_id": rid, "text": text} for rid, text in reviews]
    return (
        f"Diagnose the following {len(payload)} reviews.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=1)}"
    )
