import pytest

from review_distill.metrics import compute_metrics, latency_stats
from review_distill.schema import ASPECT_NAMES, ReviewDiagnosis


def diag(issue: str = "none", contact: bool = False, **aspects: str) -> ReviewDiagnosis:
    values = dict.fromkeys(ASPECT_NAMES, "not_mentioned") | aspects
    return ReviewDiagnosis.model_validate(
        {"aspects": values, "main_issue": issue, "needs_contact": contact}
    )


GOLD = [
    diag("none", False, delivery_time="positive"),
    diag("late_delivery", False, delivery_time="negative"),
    diag("damaged", True, packaging="negative"),
    diag("missing_item", True, order_accuracy="negative"),
]


def test_perfect_predictions():
    m = compute_metrics(GOLD, GOLD)
    assert m["n"] == 4
    assert m["json_valid_pct"] == 100
    assert m["aspects_macro_f1"] == 1
    assert set(m["aspect_f1"].values()) == {1}
    assert m["main_issue_macro_f1"] == 1
    assert m["main_issue_accuracy"] == 1
    assert m["needs_contact_f1"] == 1
    assert m["exact_match_pct"] == 100


def test_aspect_f1_is_macro_over_present_classes():
    # delivery_time gold: pos, neg, nm, nm. Predict all not_mentioned.
    preds = [diag("none"), diag("late_delivery"), GOLD[2], GOLD[3]]
    m = compute_metrics(GOLD, preds)
    # positive F1 0, negative F1 0, not_mentioned: P=2/4, R=1 -> F1=2/3; mean = 2/9
    assert m["aspect_f1"]["delivery_time"] == pytest.approx(2 / 9, abs=1e-4)
    # packaging only has classes neg and nm (positive absent) -> perfect stays 1
    assert m["aspect_f1"]["packaging"] == 1


def test_main_issue_metrics():
    preds = [*GOLD[:3], diag("wrong_item", True, order_accuracy="negative")]
    m = compute_metrics(GOLD, preds)
    assert m["main_issue_accuracy"] == 0.75
    # classes present: none, late, damaged, missing (F1 0), wrong (F1 0) -> 3/5
    assert m["main_issue_macro_f1"] == 0.6
    assert m["main_issue_f1_per_class"] == {
        "none": 1,
        "late_delivery": 1,
        "damaged": 1,
        "missing_item": 0,
    }
    assert m["main_issue_support"]["damaged"] == 1


def test_needs_contact_f1_on_positive_class():
    preds = [GOLD[0], diag("late_delivery", True, delivery_time="negative"), GOLD[2], GOLD[3]]
    m = compute_metrics(GOLD, preds)
    # TP 2, FP 1, FN 0 -> P 2/3, R 1, F1 0.8
    assert m["needs_contact_precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["needs_contact_recall"] == 1
    assert m["needs_contact_f1"] == 0.8


def test_invalid_predictions_count_as_wrong():
    preds = [GOLD[0], GOLD[1], None, None]
    m = compute_metrics(GOLD, preds)
    assert m["json_valid_pct"] == 50
    assert m["main_issue_accuracy"] == 0.5
    assert m["needs_contact_recall"] == 0
    assert m["needs_contact_f1"] == 0
    assert m["exact_match_pct"] == 50
    assert m["aspect_f1"]["packaging"] < 1


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="predictions"):
        compute_metrics(GOLD, GOLD[:2])


def test_latency_stats():
    s = latency_stats([0.01, 0.02, 0.03, 0.04])
    assert s["latency_ms_mean"] == 25
    assert s["latency_ms_p50"] == 25
    assert 38 < s["latency_ms_p95"] <= 40
