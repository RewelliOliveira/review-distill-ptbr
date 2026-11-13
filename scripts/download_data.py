"""Download the Olist reviews dataset from Kaggle into data/raw/.

The dataset is public, so Kaggle credentials are optional (KAGGLE_USERNAME /
KAGGLE_KEY or ~/.kaggle/kaggle.json are used if present).
"""

import argparse
import shutil
from pathlib import Path

import kagglehub
from dotenv import load_dotenv

from review_distill.data import KAGGLE_DATASET, RAW_DIR, REVIEWS_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    parser.add_argument("--all", action="store_true", help="copy every CSV, not only reviews")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    load_dotenv()
    src_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET, force_download=args.force))
    files = sorted(src_dir.glob("*.csv")) if args.all else [src_dir / REVIEWS_FILE]

    args.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f"{f.name} not found in {src_dir}")
        shutil.copy2(f, args.out / f.name)
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB -> {args.out / f.name}")


if __name__ == "__main__":
    main()
