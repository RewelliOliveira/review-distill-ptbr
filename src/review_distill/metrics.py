"""Evaluation metrics shared by every model (same function, same gold set).

A prediction is a ReviewDiagnosis, or None when the model output was not valid JSON for
the schema. Invalid predictions are counted as wrong in every field (they never match a
gold value), so a model cannot improve its scores by failing to answer.

- Aspects: F1 per aspect, macro over the three classes (positive, negative,
  not_mentioned); `aspects_macro_f1` is the mean over the six aspects.
- main_issue: macro F1 over the ten classes, plus accuracy.
- needs_contact: F1 of the positive class (customer needs contact).
- Classes absent from both gold and predictions are left out of macro averages
  (instead of counting as F1 = 0 or 1).
"""

import statistics
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from review_distill.schema import ASPECT_NAMES, MainIssue, ReviewDiagnosis, Sentiment

INVALID = "__invalid__"
SENTIMENTS = [s.value for s in Sentiment]
ISSUES = [i.value for i in MainIssue]


def _macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: list[str]) -> float:
    present = [label for label in labels if label in y_true or label in y_pred]
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def _field(preds: Sequence[ReviewDiagnosis | None], get) -> list[str]:
    return [INVALID if p is None else str(get(p)) for p in preds]


def _aspect(name: str):
    return lambda d: getattr(d.aspects, name).value


def _round(value: float) -> float:
    return round(float(value), 4)


def compute_metrics(
    gold: Sequence[ReviewDiagnosis], preds: Sequence[ReviewDiagnosis | None]
) -> dict[str, Any]:
    """Quality metrics of `preds` against `gold` (same order, same length)."""
    if len(gold) != len(preds):
        raise ValueError(f"{len(gold)} gold labels but {len(preds)} predictions")
    if not gold:
        raise ValueError("no examples to evaluate")

    aspect_f1 = {}
    for name in ASPECT_NAMES:
        y_true, y_pred = _field(gold, _aspect(name)), _field(preds, _aspect(name))
        aspect_f1[name] = _round(_macro_f1(y_true, y_pred, SENTIMENTS))

    issue_true = _field(gold, lambda d: d.main_issue.value)
    issue_pred = _field(preds, lambda d: d.main_issue.value)
    issue_f1 = f1_score(issue_true, issue_pred, labels=ISSUES, average=None, zero_division=0)
    issue_support = {i: issue_true.count(i) for i in ISSUES}

    contact_true = _field(gold, lambda d: d.needs_contact)
    contact_pred = _field(preds, lambda d: d.needs_contact)
    contact_kwargs = {"labels": ["True"], "average": "macro", "zero_division": 0}

    n_valid = sum(p is not None for p in preds)
    exact = sum(p is not None and p == g for g, p in zip(gold, preds, strict=True))
    return {
        "n": len(gold),
        "json_valid_pct": _round(100 * n_valid / len(gold)),
        "aspects_macro_f1": _round(statistics.mean(aspect_f1.values())),
        "aspect_f1": aspect_f1,
        "main_issue_macro_f1": _round(_macro_f1(issue_true, issue_pred, ISSUES)),
        "main_issue_accuracy": _round(accuracy_score(issue_true, issue_pred)),
        "main_issue_f1_per_class": {
            i: _round(f) for i, f in zip(ISSUES, issue_f1, strict=True) if issue_support[i]
        },
        "main_issue_support": {i: n for i, n in issue_support.items() if n},
        "needs_contact_f1": _round(f1_score(contact_true, contact_pred, **contact_kwargs)),
        "needs_contact_precision": _round(
            precision_score(contact_true, contact_pred, **contact_kwargs)
        ),
        "needs_contact_recall": _round(recall_score(contact_true, contact_pred, **contact_kwargs)),
        "exact_match_pct": _round(100 * exact / len(gold)),
    }


def latency_stats(seconds_per_review: Sequence[float]) -> dict[str, float]:
    """Mean, median and p95 latency in milliseconds per review."""
    if not seconds_per_review:
        raise ValueError("no latencies")
    ms = np.asarray(seconds_per_review, dtype=float) * 1000
    return {
        "latency_ms_mean": _round(ms.mean()),
        "latency_ms_p50": _round(np.percentile(ms, 50)),
        "latency_ms_p95": _round(np.percentile(ms, 95)),
    }
