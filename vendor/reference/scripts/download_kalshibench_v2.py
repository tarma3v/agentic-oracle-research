"""
Download KalshiBench v2 from Hugging Face once and save it to disk for offline use.

Why this exists:
- `datasets.load_dataset()` already caches in ~/.cache by default, but that cache isn't always convenient
  to share, move, or keep inside this repo.
- This script creates a portable, repo-local copy via `Dataset.save_to_disk()`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "2084Collective/kalshibench-v2"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download KalshiBench v2 and save to disk for offline loading"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/kalshibench-v2"),
        help="Output directory (default: cache/kalshibench-v2)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download/save (default: train)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    if out_dir.exists():
        if not args.force:
            raise SystemExit(
                f"Refusing to overwrite existing directory: {out_dir}\n"
                "Re-run with --force to overwrite."
            )
        # `save_to_disk()` requires an empty/non-existent directory.
        shutil.rmtree(out_dir)

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_ID} split={args.split} ...")
    ds = load_dataset(DATASET_ID, split=args.split)

    print(f"Saving to {out_dir} ...")
    ds.save_to_disk(str(out_dir))

    print("Done.")
    print(
        "Loader will auto-prefer this local copy if present.\n"
        f"- Saved: {out_dir}\n"
        "- Optional offline mode:\n"
        "  export HF_DATASETS_OFFLINE=1\n"
        "  export HF_HUB_OFFLINE=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
