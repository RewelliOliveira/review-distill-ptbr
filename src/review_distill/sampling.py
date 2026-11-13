"""Stratified sampling of reviews by score, with a fixed seed.

The gold test set is drawn first; every other sample must exclude its review_ids
(and its texts, compared case-insensitively) so it can never leak into training.
"""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from review_distill.data import LABELED_DIR, PROJECT_ROOT

GOLD_DIR = PROJECT_ROOT / "data" / "gold"
GOLD_REVIEWS_PATH = GOLD_DIR / "gold_reviews.csv"
POOL_REVIEWS_PATH = LABELED_DIR / "pool_reviews.csv"

SEED = 42
GOLD_PER_SCORE = 30
POOL_PER_SCORE = 400


def stratified_sample(
    reviews: pd.DataFrame,
    per_score: int,
    seed: int = SEED,
    exclude_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Sample `per_score` reviews for each score (1-5), excluding `exclude_ids`.

    Deterministic for a given input and seed. Raises if a score has too few reviews.
    The result is shuffled (with the same seed) so that any prefix stays roughly balanced.
    """
    excluded = set(exclude_ids)
    available = reviews[~reviews["review_id"].isin(excluded)]
    parts = []
    for score in sorted(available["score"].unique()):
        group = available[available["score"] == score]
        if len(group) < per_score:
            raise ValueError(f"score {score}: only {len(group)} reviews, need {per_score}")
        parts.append(group.sample(n=per_score, random_state=seed))
    sample = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return sample.reset_index(drop=True)


def assert_disjoint(gold: pd.DataFrame, other: pd.DataFrame) -> None:
    """Fail if `other` shares any review_id or (case-insensitive) text with the gold set."""
    shared_ids = set(gold["review_id"]) & set(other["review_id"])
    shared_texts = set(gold["text"].str.casefold()) & set(other["text"].str.casefold())
    if shared_ids or shared_texts:
        raise AssertionError(
            f"gold leakage: {len(shared_ids)} shared ids, {len(shared_texts)} shared texts"
        )


def save_sample(sample: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample[["review_id", "score", "text"]].to_csv(path, index=False)


def load_sample(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"review_id": str, "text": str})
