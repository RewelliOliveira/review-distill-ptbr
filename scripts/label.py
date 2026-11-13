"""Label a saved sample of reviews with Gemini (resumable).

Samples come from scripts/make_samples.py:
- pool: train/val pool -> data/labeled/labels.jsonl
- gold: gold test set  -> data/gold/gemini_labels.jsonl (to be reviewed by hand)

The pool is stored shuffled, so --n labels a roughly score-balanced prefix (e.g. a pilot)
and a later run with a larger --n extends it. Already labeled reviews are skipped.

Examples:
    uv run python scripts/label.py --sample pool --n 60
    uv run python scripts/label.py --sample pool
    uv run python scripts/label.py --sample gold
"""

import argparse
import logging
import os

from dotenv import load_dotenv

from review_distill.data import LABELED_DIR
from review_distill.labeling import (
    DEFAULT_MODEL,
    DEFAULT_THINKING_LEVEL,
    GeminiClient,
    LabelingConfig,
    label_reviews,
)
from review_distill.sampling import GOLD_DIR, GOLD_REVIEWS_PATH, POOL_REVIEWS_PATH, load_sample

SAMPLES = {
    # name: (sample CSV, labels JSONL, invalid JSONL)
    "pool": (POOL_REVIEWS_PATH, LABELED_DIR / "labels.jsonl", LABELED_DIR / "invalid.jsonl"),
    "gold": (
        GOLD_REVIEWS_PATH,
        GOLD_DIR / "gemini_labels.jsonl",
        GOLD_DIR / "gemini_invalid.jsonl",
    ),
}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--sample", choices=SAMPLES, required=True)
    parser.add_argument("--n", type=int, default=None, help="label only the first N reviews")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-requests", type=int, default=None, help="max requests this run")
    parser.add_argument("--rpm", type=float, default=10.0, help="max requests per minute")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--thinking-level",
        choices=["none", "minimal", "low", "medium", "high"],
        default=os.environ.get("GEMINI_THINKING_LEVEL", DEFAULT_THINKING_LEVEL),
        help='model reasoning effort ("none" = do not set it)',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry-invalid", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not os.environ.get("GOOGLE_API_KEY"):
        parser.error("GOOGLE_API_KEY is not set (see .env.example)")

    sample_path, labeled_path, invalid_path = SAMPLES[args.sample]
    if not sample_path.exists():
        parser.error(f"{sample_path} not found; run scripts/make_samples.py first")
    reviews = load_sample(sample_path)
    if args.n is not None:
        reviews = reviews.head(args.n)
    pairs = list(zip(reviews["review_id"], reviews["text"], strict=True))

    thinking_level = None if args.thinking_level == "none" else args.thinking_level
    client = GeminiClient(model_name=args.model, seed=args.seed, thinking_level=thinking_level)
    config = LabelingConfig(
        batch_size=args.batch_size,
        max_requests=args.max_requests,
        min_interval_s=60.0 / args.rpm if args.rpm > 0 else 0.0,
        max_retries=args.max_retries,
        retry_invalid=args.retry_invalid,
    )
    stats = label_reviews(
        pairs,
        client,
        labeled_path=labeled_path,
        invalid_path=invalid_path,
        requests_path=LABELED_DIR / f"requests_{args.sample}.jsonl",
        config=config,
    )
    print(
        f"requests={stats.requests} labeled={stats.labeled} invalid={stats.invalid} "
        f"skipped_cached={stats.skipped_cached} stopped={stats.stopped_reason or 'completed'}"
    )


if __name__ == "__main__":
    main()
