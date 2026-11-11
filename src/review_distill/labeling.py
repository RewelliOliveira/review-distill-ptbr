"""Batched, resumable labeling of reviews with a teacher LLM (Gemini).

Design:
- Reviews are sent in batches (default 20) per request; the model returns a list of
  objects, each with its review_id. Every item is validated individually with pydantic.
- Valid labels are appended to a JSONL cache keyed by review_id; invalid items are logged
  to a separate JSONL. Reviews present in either file are skipped on the next run, so the
  process can be interrupted and resumed at any point.
- Transient API errors (429 / 5xx) are retried with exponential backoff. If retries are
  exhausted the run stops cleanly (typically the daily quota) and can be resumed later.
"""

import json
import logging
import random
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from review_distill.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_batch_prompt
from review_distill.schema import LabeledReview, gemini_batch_response_schema

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_RAW_CHARS = 2000


class TransientAPIError(Exception):
    """Retryable API error (rate limit or server error)."""


class LabelingClient(Protocol):
    """Anything that turns (system prompt, user prompt) into raw JSON text."""

    model_name: str

    def generate_json(self, system_prompt: str, user_prompt: str) -> str: ...


class GeminiClient:
    """LabelingClient backed by google-genai with structured (schema-constrained) output."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        from google import genai
        from google.genai import types

        self.model_name = model_name
        self._client = genai.Client(api_key=api_key)
        self._config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema.model_validate(gemini_batch_response_schema()),
            temperature=temperature,
            seed=seed,
        )

    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai import errors

        config = self._config.model_copy(update={"system_instruction": system_prompt})
        try:
            response = self._client.models.generate_content(
                model=self.model_name, contents=user_prompt, config=config
            )
        except errors.APIError as e:
            if e.code in RETRYABLE_STATUS:
                raise TransientAPIError(f"{e.code}: {e.message}") from e
            raise
        return response.text or ""


@dataclass
class LabelingConfig:
    batch_size: int = 20
    max_requests: int | None = None  # cap per run (e.g. to stay within a daily quota)
    min_interval_s: float = 0.0  # minimum seconds between requests (RPM limit)
    max_retries: int = 5
    backoff_base_s: float = 2.0
    backoff_max_s: float = 60.0
    retry_invalid: bool = False  # re-send reviews previously logged as invalid


@dataclass
class LabelingStats:
    requests: int = 0
    labeled: int = 0
    invalid: int = 0
    skipped_cached: int = 0
    stopped_reason: str | None = None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- JSONL cache


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A crash mid-write can leave a truncated last line; ignore it.
                logger.warning("skipping corrupt line %d in %s", line_no, path)


def read_ids(path: Path) -> set[str]:
    return {r["review_id"] for r in read_jsonl(path) if r.get("review_id")}


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    records = list(records)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------- parsing


def parse_batch_response(
    raw: str, batch: Sequence[tuple[str, str]]
) -> tuple[list[LabeledReview], list[dict[str, Any]]]:
    """Validate a raw batch response item by item.

    Returns (valid items, invalid records). Every requested review_id ends up in exactly
    one of the two (missing ones are reported as invalid).
    """
    requested = {rid for rid, _ in batch}
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        reason, error = "unparseable_response", str(e)
        items = None
    else:
        reason, error = "not_a_list", f"got {type(items).__name__}"
    if not isinstance(items, list):
        raw_snippet = raw[:MAX_RAW_CHARS]
        return [], [
            {"review_id": rid, "reason": reason, "error": error, "raw": raw_snippet}
            for rid in sorted(requested)
        ]

    valid: list[LabeledReview] = []
    invalid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        rid = item.get("review_id") if isinstance(item, dict) else None
        rid = rid if isinstance(rid, str) else None
        if rid is not None and rid not in requested:
            invalid.append({"review_id": None, "reason": "unknown_id", "raw": item})
            continue
        if rid is not None and rid in seen:
            invalid.append({"review_id": None, "reason": "duplicate_id", "raw": item})
            continue
        try:
            labeled = LabeledReview.model_validate(item)
        except ValidationError as e:
            invalid.append(
                {
                    "review_id": rid,
                    "reason": "validation_error",
                    "error": json.dumps(e.errors(include_url=False), default=str),
                    "raw": item,
                }
            )
        else:
            valid.append(labeled)
        if rid is not None:
            seen.add(rid)

    for rid in sorted(requested - seen):
        invalid.append({"review_id": rid, "reason": "missing_in_response"})
    return valid, invalid


# ---------------------------------------------------------------- main loop


def _batched(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _call_with_retry(
    client: LabelingClient,
    prompt: str,
    config: LabelingConfig,
    sleep: Callable[[float], None],
    rng: random.Random,
) -> str:
    for attempt in range(config.max_retries + 1):
        try:
            return client.generate_json(SYSTEM_PROMPT, prompt)
        except TransientAPIError as e:
            if attempt == config.max_retries:
                raise
            delay = min(config.backoff_max_s, config.backoff_base_s * 2**attempt)
            delay *= 1 + rng.random() * 0.25  # jitter
            logger.warning("transient error (%s); retry %d in %.1fs", e, attempt + 1, delay)
            sleep(delay)
    raise AssertionError("unreachable")


def label_reviews(
    reviews: Sequence[tuple[str, str]],
    client: LabelingClient,
    labeled_path: Path,
    invalid_path: Path,
    config: LabelingConfig | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> LabelingStats:
    """Label (review_id, text) pairs, appending results to the JSONL files.

    Already-labeled reviews (and, unless retry_invalid, already-invalid ones) are skipped.
    """
    config = config or LabelingConfig()
    stats = LabelingStats()
    rng = random.Random(0)

    done = read_ids(labeled_path)
    if not config.retry_invalid:
        done |= read_ids(invalid_path)
    texts = dict(reviews)
    pending = [(rid, text) for rid, text in reviews if rid not in done]
    stats.skipped_cached = len(reviews) - len(pending)

    meta = {"model": client.model_name, "prompt_version": PROMPT_VERSION}
    last_request: float | None = None

    for batch in _batched(pending, config.batch_size):
        if config.max_requests is not None and stats.requests >= config.max_requests:
            stats.stopped_reason = "max_requests"
            break
        if last_request is not None and config.min_interval_s > 0:
            wait = config.min_interval_s - (clock() - last_request)
            if wait > 0:
                sleep(wait)
        last_request = clock()
        stats.requests += 1

        try:
            raw = _call_with_retry(client, build_batch_prompt(batch), config, sleep, rng)
        except TransientAPIError as e:
            stats.stopped_reason = f"retries_exhausted: {e}"
            stats.errors.append(str(e))
            break

        valid, invalid = parse_batch_response(raw, batch)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        append_jsonl(
            labeled_path,
            (
                {
                    "review_id": v.review_id,
                    "text": texts[v.review_id],
                    **v.diagnosis().model_dump(mode="json"),
                    **meta,
                    "labeled_at": now,
                }
                for v in valid
            ),
        )
        append_jsonl(invalid_path, ({**r, **meta, "logged_at": now} for r in invalid))
        stats.labeled += len(valid)
        stats.invalid += sum(1 for r in invalid if r["review_id"] is not None)
        logger.info(
            "request %d: %d valid, %d invalid (total labeled this run: %d)",
            stats.requests,
            len(valid),
            len(invalid),
            stats.labeled,
        )

    return stats
