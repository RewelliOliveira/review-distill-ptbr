import json
from collections.abc import Callable
from pathlib import Path

import pytest

from review_distill.labeling import (
    LabelingConfig,
    TransientAPIError,
    append_jsonl,
    label_reviews,
    parse_batch_response,
    read_ids,
    read_jsonl,
)
from review_distill.schema import ASPECT_NAMES

ASPECTS = dict.fromkeys(ASPECT_NAMES, "not_mentioned")


def item(rid: str, **overrides) -> dict:
    return {
        "review_id": rid,
        "aspects": dict(ASPECTS),
        "main_issue": "none",
        "needs_contact": False,
        **overrides,
    }


def ids_in_prompt(prompt: str) -> list[str]:
    return [r["review_id"] for r in json.loads(prompt[prompt.index("[") :])]


class FakeClient:
    """Returns a valid label for every review in the prompt, unless a responder is given."""

    model_name = "fake-model"

    def __init__(self, responder: Callable[[list[str], int], str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.responder = responder

    def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        ids = ids_in_prompt(user_prompt)
        self.calls.append(ids)
        if self.responder:
            return self.responder(ids, len(self.calls))
        return json.dumps([item(i) for i in ids])


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s

    def __call__(self) -> float:
        return self.now


REVIEWS = [(f"r{i}", f"texto {i}") for i in range(7)]


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "labels.jsonl", tmp_path / "invalid.jsonl"


def run(reviews, client, paths, **cfg):
    clock = FakeClock()
    stats = label_reviews(
        reviews, client, *paths, LabelingConfig(**cfg), sleep=clock.sleep, clock=clock
    )
    return stats, clock


# ---------------------------------------------------------------- parsing


def test_parse_all_valid() -> None:
    batch = [("a", "x"), ("b", "y")]
    valid, invalid = parse_batch_response(json.dumps([item("a"), item("b")]), batch)
    assert [v.review_id for v in valid] == ["a", "b"]
    assert invalid == []


def test_parse_validates_each_item_individually() -> None:
    batch = [("a", "x"), ("b", "y"), ("c", "z")]
    raw = json.dumps([item("a"), item("b", main_issue="bogus"), item("zzz")])
    valid, invalid = parse_batch_response(raw, batch)
    assert [v.review_id for v in valid] == ["a"]
    reasons = {(r["review_id"], r["reason"]) for r in invalid}
    assert reasons == {
        ("b", "validation_error"),
        (None, "unknown_id"),
        ("c", "missing_in_response"),
    }


def test_parse_duplicate_id_keeps_first() -> None:
    raw = json.dumps([item("a"), item("a", needs_contact=True)])
    valid, invalid = parse_batch_response(raw, [("a", "x")])
    assert len(valid) == 1 and valid[0].needs_contact is False
    assert invalid[0]["reason"] == "duplicate_id"


@pytest.mark.parametrize(
    ("raw", "reason"), [("not json", "unparseable_response"), ('{"a": 1}', "not_a_list")]
)
def test_parse_bad_response_marks_whole_batch_invalid(raw: str, reason: str) -> None:
    valid, invalid = parse_batch_response(raw, [("a", "x"), ("b", "y")])
    assert valid == []
    assert {(r["review_id"], r["reason"]) for r in invalid} == {("a", reason), ("b", reason)}


# ---------------------------------------------------------------- cache


def test_read_jsonl_ignores_truncated_line(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    append_jsonl(p, [{"review_id": "a"}])
    with p.open("a", encoding="utf-8") as f:
        f.write('{"review_id": "b", "aspe')  # crash mid-write
    assert read_ids(p) == {"a"}


# ---------------------------------------------------------------- loop


def test_batches_and_writes_labels(paths) -> None:
    client = FakeClient()
    stats, _ = run(REVIEWS, client, paths, batch_size=3)
    assert [len(c) for c in client.calls] == [3, 3, 1]
    assert stats.requests == 3 and stats.labeled == 7 and stats.invalid == 0
    rows = list(read_jsonl(paths[0]))
    assert [r["review_id"] for r in rows] == [r for r, _ in REVIEWS]
    assert rows[0]["text"] == "texto 0"
    assert rows[0]["model"] == "fake-model" and rows[0]["prompt_version"]
    assert set(rows[0]) >= {"aspects", "main_issue", "needs_contact"}


def test_resume_skips_cached(paths) -> None:
    run(REVIEWS[:4], FakeClient(), paths, batch_size=2)
    client = FakeClient()
    stats, _ = run(REVIEWS, client, paths, batch_size=2)
    assert stats.skipped_cached == 4
    assert client.calls == [["r4", "r5"], ["r6"]]
    assert len(read_ids(paths[0])) == 7


def test_invalid_logged_separately_and_skipped_unless_retry(paths) -> None:
    def responder(ids, _):
        return json.dumps([item(i, main_issue="bogus" if i == "r1" else "none") for i in ids])

    stats, _ = run(REVIEWS[:3], FakeClient(responder), paths, batch_size=3)
    assert stats.labeled == 2 and stats.invalid == 1
    assert read_ids(paths[0]) == {"r0", "r2"}
    assert read_ids(paths[1]) == {"r1"}

    client = FakeClient()
    run(REVIEWS[:3], client, paths)
    assert client.calls == []  # invalid one is not re-sent by default

    run(REVIEWS[:3], client, paths, retry_invalid=True)
    assert client.calls == [["r1"]]
    assert read_ids(paths[0]) == {"r0", "r1", "r2"}


def test_retries_429_with_exponential_backoff(paths) -> None:
    def responder(ids, n_call):
        if n_call <= 2:
            raise TransientAPIError("429: rate limited")
        return json.dumps([item(i) for i in ids])

    stats, clock = run(REVIEWS[:2], FakeClient(responder), paths, backoff_base_s=1.0)
    assert stats.labeled == 2 and stats.requests == 1
    assert len(clock.sleeps) == 2
    assert 1.0 <= clock.sleeps[0] <= 1.25 and 2.0 <= clock.sleeps[1] <= 2.5


def test_stops_cleanly_when_retries_exhausted(paths) -> None:
    def responder(ids, n_call):
        if n_call == 1:
            return json.dumps([item(i) for i in ids])
        raise TransientAPIError("429: quota exceeded")

    stats, _ = run(REVIEWS, FakeClient(responder), paths, batch_size=2, max_retries=2)
    assert stats.labeled == 2
    assert stats.stopped_reason.startswith("retries_exhausted")
    assert read_ids(paths[0]) == {"r0", "r1"}
    assert not paths[1].exists()  # rate-limited reviews are not marked invalid


def test_request_log_records_attempts_429s_and_latency(paths, tmp_path) -> None:
    requests_path = tmp_path / "requests.jsonl"
    clock = FakeClock()

    def responder(ids, n_call):
        if n_call == 1:
            raise TransientAPIError("429: rate limited")
        if n_call == 2:
            raise TransientAPIError("503: unavailable")
        clock.now += 3.0  # the successful call takes 3s
        return json.dumps([item(i) for i in ids])

    client = FakeClient(responder)
    client.last_usage = {"prompt_tokens": 10, "output_tokens": 5, "thinking_tokens": 0}
    config = LabelingConfig(backoff_base_s=1.0)
    label_reviews(
        REVIEWS[:2], client, *paths, config, clock.sleep, clock, requests_path=requests_path
    )

    [log] = list(read_jsonl(requests_path))
    assert log["ok"] is True and log["n_valid"] == 2
    assert log["attempts"] == 3 and log["rate_limited"] == 1 and log["other_errors"] == 1
    assert log["last_call_s"] == 3.0
    assert log["total_s"] == round(sum(clock.sleeps) + 3.0, 2)
    assert log["thinking_tokens"] == 0 and log["model"] == "fake-model"


def test_request_log_records_exhausted_request(paths, tmp_path) -> None:
    requests_path = tmp_path / "requests.jsonl"

    def responder(ids, n_call):
        raise TransientAPIError("429: quota exceeded")

    clock = FakeClock()
    config = LabelingConfig(max_retries=1)
    label_reviews(
        REVIEWS[:2],
        FakeClient(responder),
        *paths,
        config,
        clock.sleep,
        clock,
        requests_path=requests_path,
    )
    [log] = list(read_jsonl(requests_path))
    assert log["ok"] is False and log["attempts"] == 2 and log["rate_limited"] == 2


def test_transient_error_parses_status() -> None:
    assert TransientAPIError("429: x").status == 429
    assert TransientAPIError("boom", status=503).status == 503
    assert TransientAPIError("boom").status is None


def test_non_retryable_error_propagates(paths) -> None:
    def responder(ids, _):
        raise ValueError("bad api key")

    with pytest.raises(ValueError):
        run(REVIEWS, FakeClient(responder), paths)


def test_max_requests_cap(paths) -> None:
    client = FakeClient()
    stats, _ = run(REVIEWS, client, paths, batch_size=2, max_requests=2)
    assert len(client.calls) == 2 and stats.stopped_reason == "max_requests"
    assert stats.labeled == 4


def test_min_interval_between_requests(paths) -> None:
    _, clock = run(REVIEWS, FakeClient(), paths, batch_size=3, min_interval_s=6.0)
    assert clock.sleeps == [6.0, 6.0]
