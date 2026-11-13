"""Chat format, generation and output parsing for the student model.

The student sees a short instruction plus the review text and answers with the diagnosis
as compact JSON. The same format is used for training (prompt -> completion) and inference.
Raw outputs are saved as predictions JSONL ({"review_id", "raw", "latency_s"}) and parsed
by scripts/evaluate.py, so every model goes through the same parser and metrics.

transformers/torch are imported lazily: this module also runs on the CPU-only setup.
"""

import json
import time
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from review_distill.schema import ReviewDiagnosis

STUDENT_PROMPT_VERSION = "s1"
STUDENT_SYSTEM_PROMPT = (
    "You extract a structured diagnosis from a Brazilian e-commerce customer review "
    "(in Portuguese). Answer only with a JSON object with the keys aspects "
    "(delivery_time, product_quality, as_described, order_accuracy, packaging, "
    "customer_service: positive | negative | not_mentioned), main_issue and needs_contact."
)

Message = dict[str, str]


def prompt_messages(text: str) -> list[Message]:
    return [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def diagnosis_to_json(diagnosis: ReviewDiagnosis) -> str:
    return json.dumps(diagnosis.model_dump(mode="json"), ensure_ascii=False)


def training_example(text: str, diagnosis: ReviewDiagnosis) -> dict[str, list[Message]]:
    """Conversational prompt-completion example (TRL computes the loss on the completion)."""
    return {
        "prompt": prompt_messages(text),
        "completion": [{"role": "assistant", "content": diagnosis_to_json(diagnosis)}],
    }


def parse_diagnosis(raw: str) -> ReviewDiagnosis | None:
    """Parse a model output into a diagnosis, or None if it is not valid for the schema.

    Tolerates surrounding text or code fences: the outermost {...} is parsed.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        return ReviewDiagnosis.model_validate(json.loads(raw[start : end + 1]))
    except (json.JSONDecodeError, ValidationError):
        return None


def prediction_record(review_id: str, raw: str, latency_s: float) -> dict[str, Any]:
    return {"review_id": review_id, "raw": raw, "latency_s": round(latency_s, 5)}


def generate_raw(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    batch_size: int = 16,
    max_new_tokens: int = 128,
) -> list[tuple[str, float]]:
    """Greedy generation for each text. Returns (raw output, latency in s per review).

    Latency is the batch wall time divided by the batch size (amortized, batched).
    """
    import torch

    tokenizer.padding_side = "left"  # decoder-only batched generation
    results: list[tuple[str, float]] = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                prompt_messages(t), tokenize=False, add_generation_prompt=True
            )
            for t in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per_review = (time.perf_counter() - start) / len(batch)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        results.extend((raw, per_review) for raw in decoded)
    return results
