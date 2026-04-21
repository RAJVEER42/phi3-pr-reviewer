#!/usr/bin/env python3
"""Phase 2.5 — format filtered pairs into Phi-3 chat-template rows, then split.

Input:  data/processed/pairs.jsonl (from build_pairs.py)
Output: data/processed/train.jsonl   (~90% of pairs)
        data/processed/val.jsonl     (~5% of pairs)
        data/processed/test.jsonl    (~5% of pairs)

Each output row carries the "messages" format SFTTrainer expects natively,
so we never hand-format chat template strings (the #1 source of train-vs-inference
token drift). The tokenizer's apply_chat_template handles templating at
training time:

    {
      "messages": [
        {"role": "user", "content": "Review this code change:\\n```diff\\n<hunk>\\n```"},
        {"role": "assistant", "content": "<comment>"}
      ],
      "repo": "...", "pr_number": 123, "file_path": "..."
    }

Split is STRATIFIED BY REPO per TASK_SPEC §5 — every repo appears in all three
splits proportional to its share. Seed is deterministic (default 42).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_INPUT = "data/processed/pairs.jsonl"
DEFAULT_OUT_DIR = "data/processed"
USER_TEMPLATE = "Review this code change:\n```diff\n{hunk}\n```"

# Train / val / test — matches TASK_SPEC §5.
DEFAULT_RATIOS = (0.90, 0.05, 0.05)


def load_pairs(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def format_row(pair: dict) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": USER_TEMPLATE.format(hunk=pair["diff_hunk"]),
            },
            {"role": "assistant", "content": pair["comment"]},
        ],
        "repo": pair["repo"],
        "pr_number": pair["pr_number"],
        "file_path": pair["file_path"],
    }


def stratified_split(
    rows: list[dict],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split per repo so every repo is present in all three splits."""
    rng = random.Random(seed)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)

    train, val, test = [], [], []
    for repo, items in by_repo.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        # Remainder → test. Guarantees every repo with ≥3 pairs is in all splits.
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def print_distribution(train: list[dict], val: list[dict], test: list[dict]) -> None:
    all_repos = sorted({r["repo"] for r in train + val + test})
    cnt_train = Counter(r["repo"] for r in train)
    cnt_val = Counter(r["repo"] for r in val)
    cnt_test = Counter(r["repo"] for r in test)

    print("\n=== Per-repo split distribution ===")
    print(f"{'repo':<40} {'train':>6} {'val':>5} {'test':>5}")
    print("-" * 60)
    for repo in all_repos:
        print(
            f"{repo:<40} "
            f"{cnt_train[repo]:>6} "
            f"{cnt_val[repo]:>5} "
            f"{cnt_test[repo]:>5}"
        )
    print("-" * 60)
    print(
        f"{'TOTAL':<40} "
        f"{len(train):>6} "
        f"{len(val):>5} "
        f"{len(test):>5}"
    )

    missing_val = [r for r in all_repos if cnt_val[r] == 0]
    missing_test = [r for r in all_repos if cnt_test[r] == 0]
    if missing_val or missing_test:
        print("\n  WARN: some repos absent from val/test due to small sample:")
        if missing_val:
            print(f"    missing from val:  {missing_val}")
        if missing_test:
            print(f"    missing from test: {missing_test}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    formatted = [format_row(p) for p in load_pairs(input_path)]
    print(f"Loaded {len(formatted)} formatted pairs from {input_path}")

    if not formatted:
        print("No pairs to split. Exiting.", file=sys.stderr)
        return 1

    train, val, test = stratified_split(formatted, DEFAULT_RATIOS, args.seed)
    print_distribution(train, val, test)

    for name, split in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for row in split:
                f.write(json.dumps(row) + "\n")
        print(f"  wrote {path}  ({len(split)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
