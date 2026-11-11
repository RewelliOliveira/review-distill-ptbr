"""Loading and cleaning of the Olist reviews dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"
REVIEWS_FILE = "olist_order_reviews_dataset.csv"
KAGGLE_DATASET = "olistbr/brazilian-ecommerce"


def load_raw_reviews(path: Path | None = None) -> pd.DataFrame:
    """Load the raw Olist reviews CSV, without any filtering."""
    path = path or RAW_DIR / REVIEWS_FILE
    return pd.read_csv(path, dtype={"review_id": str, "order_id": str})


def build_text(title: object, message: object) -> str:
    """Join review title and message into one normalized text ('' if both empty)."""
    parts = []
    for part in (title, message):
        if isinstance(part, str):
            part = " ".join(part.split())  # collapse \r\n and repeated whitespace
            if part:
                parts.append(part)
    return ". ".join(parts)


def prepare_reviews(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per usable review: columns review_id, score, text.

    Drops (by design, see README): reviews without text (rating only), repeated
    review_ids (Olist reuses an id across orders) and exact duplicate texts
    (dedup before splitting avoids train/test leakage).
    """
    df = raw.assign(
        text=[
            build_text(t, m)
            for t, m in zip(raw["review_comment_title"], raw["review_comment_message"], strict=True)
        ]
    )
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset="review_id", keep="first")
    df = df.drop_duplicates(subset="text", keep="first")
    return df[["review_id", "review_score", "text"]].rename(columns={"review_score": "score"})


def load_reviews(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned reviews (see `prepare_reviews`)."""
    return prepare_reviews(load_raw_reviews(path)).reset_index(drop=True)
