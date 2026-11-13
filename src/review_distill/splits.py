"""Train/validation split of the teacher-labeled pool, stratified by main_issue.

The gold test set is never part of this split: the pool was drawn excluding it, and the
split re-checks that no review_id or text is shared with the gold set.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from review_distill.data import LABELED_DIR
from review_distill.labeling import read_jsonl
from review_distill.sampling import SEED
from review_distill.schema import ReviewDiagnosis

POOL_LABELS_PATH = LABELED_DIR / "labels.jsonl"
TRAIN_PATH = LABELED_DIR / "train.jsonl"
VAL_PATH = LABELED_DIR / "val.jsonl"
VAL_FRACTION = 0.1

# Fields written to train/val JSONL: what a model needs to learn from (no labeling metadata).
SPLIT_FIELDS = ["review_id", "text", *ReviewDiagnosis.model_fields]


def load_labeled(path: Path) -> pd.DataFrame:
    """Load a labels JSONL (later lines win), validating every diagnosis with pydantic."""
    records = {r["review_id"]: r for r in read_jsonl(path)}
    for r in records.values():
        ReviewDiagnosis.model_validate({k: r[k] for k in ReviewDiagnosis.model_fields})
    return pd.DataFrame([{k: r[k] for k in SPLIT_FIELDS} for r in records.values()])


def split_train_val(
    labeled: pd.DataFrame, val_fraction: float = VAL_FRACTION, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split stratified by main_issue. Deterministic for a given input and seed."""
    train, val = train_test_split(
        labeled,
        test_size=val_fraction,
        random_state=seed,
        stratify=labeled["main_issue"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def save_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[SPLIT_FIELDS].to_json(path, orient="records", lines=True, force_ascii=False)


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_json(path, orient="records", lines=True, dtype={"review_id": str})
