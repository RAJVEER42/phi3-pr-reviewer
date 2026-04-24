#!/usr/bin/env python3
"""Phase 5b — manual rubric scorer.

Samples 50 (idx-keyed, deterministic) prediction rows from the fine-tuned
predictions JSONL, joins them with the original diff hunk from the HF test
dataset, and walks you through scoring each on three dimensions per
TASK_SPEC §6.2:

    Relevance     1=off-topic/hallucinated, 3=right area wrong issue, 5=identifies the actual issue
    Actionability 1=no concrete suggestion, 3=vague, 5=specific implementable change
    Factuality    1=wrong claim, 3=ambiguous, 5=verifiable from the diff

Resumable — saves after every row to results/rubric_scores.jsonl. Re-running
skips already-scored idxs.

Prints a summary table at the end (mean per dimension, distribution, low/high
examples for the writeup).

Usage:
    python scripts/rubric_helper.py
    python scripts/rubric_helper.py --n 50 --seed 42
    python scripts/rubric_helper.py --resume   # just continue without re-prompting picks
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

DEFAULT_PREDICTIONS = "results/finetuned_predictions.jsonl"
DEFAULT_DATASET = "Itachi-42/pr-reviews"
DEFAULT_SCORES = "results/rubric_scores.jsonl"
DEFAULT_N = 50
DEFAULT_SEED = 42

ANCHORS = {
    "relevance": (
        "  1 = Off-topic or hallucinated\n"
        "  3 = Touches the right area, misses the specific issue\n"
        "  5 = Identifies the actual issue the human flagged"
    ),
    "actionability": (
        "  1 = No concrete suggestion\n"
        "  3 = Vague (e.g., 'consider refactoring')\n"
        "  5 = Specific, implementable change"
    ),
    "factuality": (
        "  1 = Wrong claim about the code\n"
        "  3 = Ambiguous\n"
        "  5 = Verifiable from the diff alone"
    ),
}


def load_predictions(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def load_diffs_from_hf(dataset_id: str) -> dict[int, str]:
    """Return idx -> diff_hunk mapping from the HF test split."""
    from dotenv import load_dotenv
    from datasets import load_dataset

    load_dotenv(override=True)
    token = os.environ.get("HF_TOKEN")
    print(f"Loading {dataset_id} test split for diff context...")
    ds = load_dataset(dataset_id, split="test", token=token)
    diffs = {}
    for i, row in enumerate(ds):
        user_msg = next(m for m in row["messages"] if m["role"] == "user")
        diffs[i] = user_msg["content"]
    return diffs


def already_scored(scores_path: Path) -> set[int]:
    if not scores_path.exists():
        return set()
    seen = set()
    for line in scores_path.read_text().splitlines():
        if line.strip():
            try:
                seen.add(json.loads(line)["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def prompt_score(dim: str) -> int | None:
    """Prompt for 1/3/5 score. Returns None on quit."""
    while True:
        raw = input(f"  {dim} [1/3/5, q to save+quit, s to skip row]: ").strip().lower()
        if raw == "q":
            return None
        if raw == "s":
            return -1  # sentinel
        if raw in {"1", "3", "5"}:
            return int(raw)
        print(f"    invalid: enter 1, 3, 5, q, or s (got '{raw}')")


def banner(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--scores", default=DEFAULT_SCORES)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--summary-only", action="store_true",
                   help="skip prompting; just print summary of existing scores")
    args = p.parse_args()

    pred_path = Path(args.predictions)
    scores_path = Path(args.scores)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        print(f"Predictions not found: {pred_path}", file=sys.stderr)
        return 1

    rows = load_predictions(pred_path)
    print(f"Loaded {len(rows)} predictions from {pred_path}")

    rng = random.Random(args.seed)
    sampled = rng.sample(rows, min(args.n, len(rows)))
    sampled_by_idx = {r["idx"]: r for r in sampled}

    if args.summary_only:
        print_summary(scores_path, sampled_by_idx)
        return 0

    seen = already_scored(scores_path)
    todo = [r for r in sampled if r["idx"] not in seen]
    print(f"Already scored: {len(seen)} of {len(sampled)}. Remaining: {len(todo)}")
    if not todo:
        print_summary(scores_path, sampled_by_idx)
        return 0

    diffs = load_diffs_from_hf(args.dataset)

    with scores_path.open("a") as scoref:
        for n, row in enumerate(todo, 1):
            idx = row["idx"]
            diff = diffs.get(idx, "<diff unavailable>")
            banner(
                f"[{n}/{len(todo)}]  {row['repo']}#{row['pr_number']}  "
                f"({row['file_path']})  idx={idx}  "
                f"BERTScore_F1={row.get('bertscore_f1', 0):.3f}"
            )
            print("\n--- DIFF (model input) ---")
            # Skip the chat-template wrapper, show only the code
            for line in diff.splitlines():
                if line.startswith(("Review this code change:", "```diff", "```")):
                    continue
                print(f"  {line}")

            print("\n--- REFERENCE (human reviewer) ---")
            print(f"  {row['reference']}")

            print("\n--- PREDICTION (fine-tuned model) ---")
            print(f"  {row['prediction'][:600]}")
            if len(row['prediction']) > 600:
                print(f"  ... [truncated; {len(row['prediction'])} chars total]")

            print("\nScore 1=poor, 3=mid, 5=good. Anchors:")
            scored = {}
            quit_now = False
            for dim, anchors in ANCHORS.items():
                print(f"\n{dim.upper()}:")
                print(anchors)
                v = prompt_score(dim)
                if v is None:
                    quit_now = True
                    break
                if v == -1:
                    scored = None
                    break
                scored[dim] = v
            if quit_now:
                print("\nQuitting. Progress saved.")
                break
            if scored is None:
                print("Skipped.")
                continue
            notes = input("  Notes (optional, Enter to skip): ").strip()
            entry = {
                "idx": idx,
                "repo": row["repo"],
                "pr_number": row["pr_number"],
                "bertscore_f1": row.get("bertscore_f1"),
                **scored,
                "notes": notes,
            }
            scoref.write(json.dumps(entry) + "\n")
            scoref.flush()

    print_summary(scores_path, sampled_by_idx)
    return 0


def print_summary(scores_path: Path, sampled_by_idx: dict) -> None:
    if not scores_path.exists():
        print("No scores yet.")
        return
    scores = [
        json.loads(l) for l in scores_path.read_text().splitlines() if l.strip()
    ]
    if not scores:
        print("No scores yet.")
        return
    banner(f"SUMMARY  ({len(scores)} examples scored)")
    for dim in ANCHORS:
        vals = [s[dim] for s in scores if dim in s]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        dist = {1: vals.count(1), 3: vals.count(3), 5: vals.count(5)}
        print(f"  {dim:14s}  mean={mean:.2f}   dist 1/3/5: {dist[1]}/{dist[3]}/{dist[5]}")

    # Top wins (5/5/5) and bottom losses (1/1/1 or close)
    def total(s):
        return s.get("relevance", 0) + s.get("actionability", 0) + s.get("factuality", 0)

    sorted_scores = sorted(scores, key=total, reverse=True)
    print("\n  Top 3 wins (highest combined):")
    for s in sorted_scores[:3]:
        print(f"    idx={s['idx']:4d}  rel={s.get('relevance')} act={s.get('actionability')} fact={s.get('factuality')}  {s.get('repo','')}#{s.get('pr_number','')}")
    print("\n  Bottom 3 losses (lowest combined):")
    for s in sorted_scores[-3:]:
        print(f"    idx={s['idx']:4d}  rel={s.get('relevance')} act={s.get('actionability')} fact={s.get('factuality')}  {s.get('repo','')}#{s.get('pr_number','')}")

    rel_mean = sum(s["relevance"] for s in scores) / len(scores) if scores else 0
    threshold = 3.5
    print(f"\n  TASK_SPEC §7 S3 threshold: relevance ≥ {threshold} → "
          f"{'✅ PASSED' if rel_mean >= threshold else '❌ MISSED'} ({rel_mean:.2f})")
    print(f"\n  Scores file: {scores_path}")


if __name__ == "__main__":
    sys.exit(main())
