"""Label Olist reviews with Gemini (resumable).

Reviews are shuffled with a fixed seed and the first --n are targeted, so re-running with
the same seed and a larger --n extends the same sample. Already labeled reviews are skipped.

Example:
    uv run python scripts/label.py --n 500 --rpm 10 --max-requests 25
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from review_distill.data import LABELED_DIR, load_reviews
from review_distill.labeling import DEFAULT_MODEL, GeminiClient, LabelingConfig, label_reviews


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--n", type=int, required=True, help="target number of reviews in the sample"
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-requests", type=int, default=None, help="max requests this run")
    parser.add_argument("--rpm", type=float, default=10.0, help="max requests per minute")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--input", type=Path, default=None, help="raw reviews CSV")
    parser.add_argument("--out-dir", type=Path, default=LABELED_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not os.environ.get("GOOGLE_API_KEY"):
        parser.error("GOOGLE_API_KEY is not set (see .env.example)")

    reviews = load_reviews(args.input).sample(frac=1.0, random_state=args.seed).head(args.n)
    pairs = list(zip(reviews["review_id"], reviews["text"], strict=True))

    client = GeminiClient(model_name=args.model, seed=args.seed)
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
        labeled_path=args.out_dir / "labels.jsonl",
        invalid_path=args.out_dir / "invalid.jsonl",
        config=config,
    )
    print(
        f"requests={stats.requests} labeled={stats.labeled} invalid={stats.invalid} "
        f"skipped_cached={stats.skipped_cached} stopped={stats.stopped_reason or 'completed'}"
    )


if __name__ == "__main__":
    main()
