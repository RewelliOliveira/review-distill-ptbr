"""Prompts for the teacher model. Bump PROMPT_VERSION on any change to the text."""

import json
from collections.abc import Sequence

PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """\
You analyze customer reviews from a Brazilian e-commerce marketplace (written in Portuguese)
and extract a structured diagnosis for each review.

For each review, classify every aspect as "positive", "negative" or "not_mentioned":
- delivery_time: delivery speed, delays, arriving before/after the expected date.
- product_quality: quality, durability or performance of the product itself.
- as_described: whether the product matches the listing (photos, description, specs, size, color,
  authenticity).
- order_accuracy: whether the right items arrived (wrong item, missing item, incomplete order).
- packaging: packaging condition and protection.
- customer_service: the seller or customer service (responsiveness, problem solving, communication).
Use "not_mentioned" when the review does not address the aspect, even indirectly.
Generic or lukewarm words ("ótimo", "bom", "razoável", "ok", "recomendo") without a specific aspect
do not make any aspect positive or negative, except that "chegou rápido"/"antes do prazo" is
positive delivery_time.

Wrong item vs. missing item vs. not as described:
- missing_item: fewer items or parts arrived than were bought (the rest never came).
  Example: "Comprei três canecas e só vieram duas" -> order_accuracy negative,
  main_issue missing_item.
- wrong_item: a different variant arrived in place of the one ordered (other color, model, flavor,
  size). This is an order_accuracy problem only; do not also mark as_described negative.
  Example: "Pedi a camiseta azul e veio verde" -> order_accuracy negative, as_described
  not_mentioned, main_issue wrong_item.
- not_as_described: the item ordered arrived but differs from the listing (photos, specs, size,
  material) or is counterfeit. Mark as_described negative.
  Example: "A bolsa é falsificada, não é original como no anúncio" -> as_described negative,
  main_issue not_as_described.

Damage from transport or handling (broken, dented, crushed, scratched) is not a product_quality
problem: mark packaging (if the packaging is mentioned) and main_issue damaged; leave
product_quality not_mentioned unless the review also judges the product itself.
  Example: "A caixa veio amassada e a jarra chegou quebrada" -> packaging negative,
  product_quality not_mentioned, main_issue damaged.

main_issue is the single most important problem the customer reports:
none, late_delivery, not_delivered, wrong_item, missing_item, damaged, poor_quality,
not_as_described, bad_service, other.
- Use "none" when the customer reports no problem.
- "not_delivered": the product has not arrived and the delivery date has passed (or the customer
  says it is lost). If it has not arrived but is still within the delivery window, use "none".
  Example: "Ainda não chegou, mas está dentro do prazo" -> main_issue none, needs_contact false.
- "damaged": product arrived broken/defective due to transport or packaging.
- "poor_quality": product works badly or has low quality, but is the right product.
- If several problems are reported, pick the most severe in this order: not_delivered >
  wrong_item > missing_item > damaged > not_as_described > poor_quality > bad_service >
  late_delivery > other.

needs_contact is true when the customer has a concrete problem that is still unresolved, namely:
- the product was not delivered and the delivery date has passed;
- a wrong, missing, damaged or not-as-described item (even if the customer asks for nothing);
- the customer asks for a refund, exchange, cancellation or a reply, or says a request is pending.
needs_contact is false otherwise: no problem, a problem already resolved, only a delay that ended
with the delivery, poor quality without any request, or vague negative opinion without a concrete
problem.
  Example: "Já abri chamado pedindo estorno e até agora nada" -> needs_contact true.
  Example: "Não gostei, esperava mais" -> main_issue poor_quality or other, needs_contact false.

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
