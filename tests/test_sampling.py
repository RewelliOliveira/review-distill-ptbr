import pandas as pd
import pytest

from review_distill.sampling import (
    assert_disjoint,
    load_sample,
    save_sample,
    stratified_sample,
)


@pytest.fixture
def reviews() -> pd.DataFrame:
    rows = [
        {"review_id": f"r{score}_{i}", "score": score, "text": f"texto {score} {i}"}
        for score in range(1, 6)
        for i in range(20)
    ]
    return pd.DataFrame(rows)


def test_stratified_sample_balances_scores(reviews):
    sample = stratified_sample(reviews, per_score=4)
    assert len(sample) == 20
    assert sample["score"].value_counts().to_dict() == dict.fromkeys(range(1, 6), 4)
    assert sample["review_id"].is_unique


def test_stratified_sample_is_deterministic(reviews):
    a = stratified_sample(reviews, per_score=4, seed=7)
    b = stratified_sample(reviews, per_score=4, seed=7)
    c = stratified_sample(reviews, per_score=4, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["review_id"]) != list(c["review_id"])


def test_stratified_sample_excludes_ids(reviews):
    gold = stratified_sample(reviews, per_score=4)
    pool = stratified_sample(reviews, per_score=16, exclude_ids=gold["review_id"])
    assert not set(gold["review_id"]) & set(pool["review_id"])
    assert_disjoint(gold, pool)


def test_stratified_sample_raises_when_score_too_small(reviews):
    with pytest.raises(ValueError, match="score"):
        stratified_sample(reviews, per_score=21)


def test_assert_disjoint_detects_shared_text_ignoring_case():
    gold = pd.DataFrame({"review_id": ["a"], "score": [1], "text": ["Não chegou"]})
    other = pd.DataFrame({"review_id": ["b"], "score": [1], "text": ["NÃO CHEGOU"]})
    with pytest.raises(AssertionError, match="leakage"):
        assert_disjoint(gold, other)


def test_save_and_load_roundtrip(reviews, tmp_path):
    path = tmp_path / "s.csv"
    sample = stratified_sample(reviews, per_score=2)
    save_sample(sample, path)
    pd.testing.assert_frame_equal(load_sample(path), sample)
