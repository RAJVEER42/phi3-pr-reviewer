#!/usr/bin/env python3
"""Phase 2.6 — push the train/val/test JSONLs to Hugging Face Hub as a DatasetDict.

Reads .env for HF_TOKEN. Loads the three JSONLs produced by format_split.py,
wraps them as a DatasetDict, and pushes to the Hub under the repo ID given
on the CLI (--repo-id your-username/pr-reviews). Private by default —
TASK_SPEC §4 rule 10 aside, there's no need to publish raw OSS review
comments under your account before the project is presentable.

Usage:
    python scripts/push_dataset.py --repo-id your-username/pr-reviews
    python scripts/push_dataset.py --repo-id your-username/pr-reviews --public
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi

DEFAULT_SPLIT_DIR = "data/processed"
SPLITS = ("train", "val", "test")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF Hub dataset repo id, e.g. your-username/pr-reviews",
    )
    parser.add_argument(
        "--split-dir",
        default=DEFAULT_SPLIT_DIR,
        help="dir containing train.jsonl / val.jsonl / test.jsonl",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="make the dataset public (default: private)",
    )
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing — add it to .env", file=sys.stderr)
        return 1

    split_dir = Path(args.split_dir)
    missing = [s for s in SPLITS if not (split_dir / f"{s}.jsonl").exists()]
    if missing:
        print(f"Missing split files: {missing} in {split_dir}", file=sys.stderr)
        print("Run scripts/format_split.py first.", file=sys.stderr)
        return 1

    splits = {}
    for s in SPLITS:
        path = split_dir / f"{s}.jsonl"
        ds = Dataset.from_json(str(path))
        print(f"  {s:<5}  {len(ds):>5} rows  ({path})")
        splits[s] = ds

    ds_dict = DatasetDict(splits)

    # HF's DatasetDict uses "validation" not "val" by convention — rename.
    ds_dict["validation"] = ds_dict.pop("val")

    print(f"\nPushing to {args.repo_id} (private={not args.public})...")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    ds_dict.push_to_hub(args.repo_id, token=token, private=not args.public)
    print(f"Done. https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
