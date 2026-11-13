"""Split the teacher-labeled pool into train (90%) and validation (10%).

Stratified by main_issue with a fixed seed. Reads data/labeled/labels.jsonl and writes
data/labeled/train.jsonl and data/labeled/val.jsonl (not versioned; reproducible).
The gold set is the test set and is never touched here, except to check for leakage.

Example:
    uv run python scripts/make_splits.py
"""

import argparse
from pathlib import Path

from review_distill.sampling import GOLD_REVIEWS_PATH, SEED, assert_disjoint, load_sample
from review_distill.splits import (
    POOL_LABELS_PATH,
    TRAIN_PATH,
    VAL_FRACTION,
    VAL_PATH,
    load_labeled,
    save_split,
    split_train_val,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=POOL_LABELS_PATH)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    labeled = load_labeled(args.labels)
    assert_disjoint(load_sample(GOLD_REVIEWS_PATH), labeled)
    train, val = split_train_val(labeled, args.val_fraction, args.seed)

    save_split(train, TRAIN_PATH)
    save_split(val, VAL_PATH)
    for name, df, path in (("train", train, TRAIN_PATH), ("val", val, VAL_PATH)):
        counts = df["main_issue"].value_counts().to_dict()
        print(f"{name}: {len(df)} reviews -> {path}\n  {counts}")


if __name__ == "__main__":
    main()
