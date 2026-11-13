"""Draw the gold test set and the train/val pool, both stratified by score.

The gold set (default 30 per score = 150) is drawn first and saved to data/gold/
(versioned). The pool (default 400 per score = 2,000) is drawn from the remaining
reviews and saved to data/labeled/ (not versioned; reproducible from the seed).

Example:
    uv run python scripts/make_samples.py
"""

import argparse
from pathlib import Path

from review_distill.data import load_reviews
from review_distill.sampling import (
    GOLD_PER_SCORE,
    GOLD_REVIEWS_PATH,
    POOL_PER_SCORE,
    POOL_REVIEWS_PATH,
    SEED,
    assert_disjoint,
    save_sample,
    stratified_sample,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-per-score", type=int, default=GOLD_PER_SCORE)
    parser.add_argument("--pool-per-score", type=int, default=POOL_PER_SCORE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--input", type=Path, default=None, help="raw reviews CSV")
    args = parser.parse_args()

    reviews = load_reviews(args.input)
    gold = stratified_sample(reviews, args.gold_per_score, seed=args.seed)
    pool = stratified_sample(
        reviews, args.pool_per_score, seed=args.seed, exclude_ids=gold["review_id"]
    )
    assert_disjoint(gold, pool)

    save_sample(gold, GOLD_REVIEWS_PATH)
    save_sample(pool, POOL_REVIEWS_PATH)
    for name, df, path in (("gold", gold, GOLD_REVIEWS_PATH), ("pool", pool, POOL_REVIEWS_PATH)):
        counts = df["score"].value_counts().sort_index().to_dict()
        print(f"{name}: {len(df)} reviews {counts} -> {path}")


if __name__ == "__main__":
    main()
