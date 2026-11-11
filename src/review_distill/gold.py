"""Manual review of the gold test set.

Gemini pre-labels the gold reviews; a human then accepts or edits each label. Reviewed
labels are appended to a JSONL file (one record per decision, last one wins), so the
review can be interrupted and resumed, and any review can be redone later.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from review_distill.labeling import append_jsonl, read_jsonl
from review_distill.sampling import GOLD_DIR
from review_distill.schema import ASPECT_NAMES, MainIssue, ReviewDiagnosis, Sentiment

GEMINI_GOLD_PATH = GOLD_DIR / "gemini_labels.jsonl"
GOLD_LABELS_PATH = GOLD_DIR / "gold_labels.jsonl"

SENTIMENT_KEYS = {"p": Sentiment.POSITIVE, "n": Sentiment.NEGATIVE, "-": Sentiment.NOT_MENTIONED}
SENTIMENT_SHORT = {v: k for k, v in SENTIMENT_KEYS.items()}
ISSUES = list(MainIssue)
BOOL_KEYS = {"y": True, "n": False}

Ask = Callable[[str], str]


def diagnosis_from_record(record: dict[str, Any]) -> ReviewDiagnosis:
    fields = {k: record[k] for k in ReviewDiagnosis.model_fields}
    return ReviewDiagnosis.model_validate(fields)


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Map review_id -> latest record in a labels JSONL (later lines override earlier)."""
    return {r["review_id"]: r for r in read_jsonl(path)}


def format_diagnosis(d: ReviewDiagnosis) -> str:
    width = max(map(len, ASPECT_NAMES))
    lines = [f"  {name:<{width}}  {getattr(d.aspects, name).value}" for name in ASPECT_NAMES]
    lines.append(f"  {'main_issue':<{width}}  {d.main_issue.value}")
    lines.append(f"  {'needs_contact':<{width}}  {d.needs_contact}")
    return "\n".join(lines)


def _ask_choice(ask: Ask, prompt: str, choices: dict[str, Any], current: Any) -> Any:
    """Ask until the answer is empty (keep `current`) or one of `choices`."""
    while True:
        answer = ask(prompt).strip().lower()
        if not answer:
            return current
        if answer in choices:
            return choices[answer]


def edit_diagnosis(d: ReviewDiagnosis, ask: Ask) -> ReviewDiagnosis:
    """Walk through every field; an empty answer keeps the current value."""
    data = d.model_dump(mode="json")
    for name in ASPECT_NAMES:
        current = Sentiment(data["aspects"][name])
        prompt = f"{name} [{SENTIMENT_SHORT[current]}] (p/n/-): "
        data["aspects"][name] = _ask_choice(ask, prompt, SENTIMENT_KEYS, current).value

    issue_menu = "  ".join(f"{i}={issue.value}" for i, issue in enumerate(ISSUES))
    current_issue = MainIssue(data["main_issue"])
    issue_choices = {str(i): issue for i, issue in enumerate(ISSUES)}
    issue_choices |= {issue.value: issue for issue in ISSUES}
    prompt = f"{issue_menu}\nmain_issue [{ISSUES.index(current_issue)}]: "
    data["main_issue"] = _ask_choice(ask, prompt, issue_choices, current_issue).value

    current_contact = data["needs_contact"]
    prompt = f"needs_contact [{'y' if current_contact else 'n'}] (y/n): "
    data["needs_contact"] = _ask_choice(ask, prompt, BOOL_KEYS, current_contact)
    return ReviewDiagnosis.model_validate(data)


def gold_record(
    review_id: str,
    text: str,
    score: int,
    diagnosis: ReviewDiagnosis,
    teacher: ReviewDiagnosis,
    teacher_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "review_id": review_id,
        "text": text,
        "score": int(score),
        **diagnosis.model_dump(mode="json"),
        "changed": diagnosis != teacher,
        "teacher_model": teacher_meta.get("model"),
        "teacher_prompt_version": teacher_meta.get("prompt_version"),
        "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def save_gold_record(record: dict[str, Any], path: Path = GOLD_LABELS_PATH) -> None:
    append_jsonl(path, [record])
