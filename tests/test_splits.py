import json

import pandas as pd
import pytest

from review_distill.schema import ASPECT_NAMES, MainIssue
from review_distill.splits import load_labeled, load_split, save_split, split_train_val

ISSUES = [MainIssue.NONE, MainIssue.LATE_DELIVERY, MainIssue.DAMAGED]


def record(rid: str, issue: str) -> dict:
    return {
        "review_id": rid,
        "text": f"texto {rid}",
        "aspects": dict.fromkeys(ASPECT_NAMES, "not_mentioned"),
        "main_issue": issue,
        "needs_contact": issue != "none",
        "model": "fake",
    }


@pytest.fixture
def labeled() -> pd.DataFrame:
    rows = [record(f"{issue}_{i}", issue.value) for issue in ISSUES for i in range(20)]
    return pd.DataFrame(rows).drop(columns="model")


def test_split_sizes_and_stratification(labeled):
    train, val = split_train_val(labeled, val_fraction=0.1)
    assert len(train) == 54 and len(val) == 6
    assert val["main_issue"].value_counts().to_dict() == {i.value: 2 for i in ISSUES}
    assert not set(train["review_id"]) & set(val["review_id"])


def test_split_is_deterministic(labeled):
    a, _ = split_train_val(labeled, seed=1)
    b, _ = split_train_val(labeled, seed=1)
    c, _ = split_train_val(labeled, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["review_id"]) != list(c["review_id"])


def test_load_labeled_keeps_last_and_drops_metadata(tmp_path):
    path = tmp_path / "labels.jsonl"
    rows = [record("a", "none"), record("a", "damaged"), record("b", "none")]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    df = load_labeled(path)
    assert list(df["review_id"]) == ["a", "b"]
    assert df.loc[0, "main_issue"] == "damaged"
    assert "model" not in df.columns


def test_load_labeled_rejects_invalid_label(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(record("a", "bogus")), encoding="utf-8")
    with pytest.raises(ValueError):
        load_labeled(path)


def test_save_and_load_roundtrip(labeled, tmp_path):
    path = tmp_path / "train.jsonl"
    save_split(labeled, path)
    loaded = load_split(path)
    assert loaded.to_dict("records") == labeled.to_dict("records")
