#!/usr/bin/env python3
"""Phase 2.3 + 2.4 — filter raw PR scrapes into (hunk, comment) training pairs.

Reads data/raw/*.jsonl (written by scrape_prs.py), explodes each PR into one
candidate per inline review comment, applies the 10 filter rules from
TASK_SPEC §4, and writes the surviving pairs to data/processed/pairs.jsonl.

Filter order (cheap → expensive):
    F7   comment has a file path
    F7b  comment author != PR author  (drops self-replies masquerading as reviews)
    F1   Python (.py) files only
    F6   PR has a merge date (defensive; scraper already enforces)
    F5   human author (bot blocklist)
    F4   comment length  20 ≤ chars ≤ 500
    F3   hunk size       5 ≤ lines ≤ 300
    F8   no external references (#1234, "as discussed", "per the RFC", ...)
    F9   dedupe on (normalized_comment, user)
    F10  per-repo cap at 15% of final total

Output: one JSONL row per (hunk, comment) pair with repo metadata, plus a
survival-report table printed to stdout and a 20-row sample file for
eyeball inspection.

Usage:
    python scripts/build_pairs.py
    python scripts/build_pairs.py --raw-dir data/raw --out-file data/processed/pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_OUT = "data/processed/pairs.jsonl"
DEFAULT_SAMPLE = "data/processed/sample_pairs.jsonl"

MIN_COMMENT_CHARS = 20
MAX_COMMENT_CHARS = 500
MIN_HUNK_LINES = 5
MAX_HUNK_LINES = 300
PER_REPO_CAP_FRACTION = 0.15

# F5 — bot blocklist (matches user login, case-insensitive). Anything ending
# in "[bot]" is treated as a bot regardless of stem.
BOT_BLOCKLIST = {
    "dependabot",
    "pre-commit-ci",
    "codecov",
    "codecov-io",
    "github-actions",
    "renovate",
    "mergify",
    "sonarcloud",
    "allcontributors",
}

# F8 — patterns that indicate the comment references context the model can't
# verify from the diff alone. Matching any means drop.
EXTERNAL_REF_PATTERNS = [
    re.compile(r"(?<!\w)#\d+\b"),
    re.compile(r"\bGH-\d+\b"),
    re.compile(r"\bas discussed\b", re.IGNORECASE),
    re.compile(r"\bper the (RFC|proposal|spec|design)\b", re.IGNORECASE),
    re.compile(r"\boffline\b", re.IGNORECASE),
    re.compile(r"\bsee (the )?(doc|docs|RFC|proposal)\b", re.IGNORECASE),
]


def is_bot(user: str | None) -> bool:
    """F5 predicate. None author is treated as bot (defensive)."""
    if not user:
        return True
    low = user.lower()
    if low.endswith("[bot]"):
        return True
    stem = low.removesuffix("[bot]")
    return stem in BOT_BLOCKLIST


def has_external_ref(text: str) -> bool:
    """F8 predicate."""
    return any(p.search(text) for p in EXTERNAL_REF_PATTERNS)


def normalize_for_dedupe(text: str) -> str:
    """Lowercase, collapse whitespace, strip — for F9 dedupe key."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_raw(raw_dir: Path) -> Iterator[dict]:
    for jsonl in sorted(raw_dir.glob("*.jsonl")):
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def explode_to_candidates(pr_rows: Iterable[dict]) -> list[dict]:
    """One PR → N candidates, one per inline review comment."""
    out = []
    for pr in pr_rows:
        for c in pr.get("review_comments", []):
            out.append(
                {
                    "repo": pr["repo"],
                    "pr_number": pr["pr_number"],
                    "pr_merged_at": pr.get("merged_at"),
                    "pr_author": pr.get("author"),
                    "file_path": c.get("path") or "",
                    "diff_hunk": c.get("diff_hunk") or "",
                    "comment": c.get("body") or "",
                    "comment_user": c.get("user"),
                }
            )
    return out


def apply_filter(
    name: str,
    candidates: list[dict],
    pred: Callable[[dict], bool],
    counts: dict[str, dict],
) -> list[dict]:
    before = len(candidates)
    kept = [c for c in candidates if pred(c)]
    counts[name] = {"before": before, "after": len(kept), "dropped": before - len(kept)}
    return kept


def dedupe(candidates: list[dict], counts: dict[str, dict]) -> list[dict]:
    before = len(candidates)
    seen: set[tuple[str, str | None]] = set()
    kept = []
    for c in candidates:
        key = (normalize_for_dedupe(c["comment"]), c["comment_user"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    counts["F9_dedupe"] = {
        "before": before,
        "after": len(kept),
        "dropped": before - len(kept),
    }
    return kept


def apply_repo_cap(
    candidates: list[dict],
    counts: dict[str, dict],
    rng: random.Random,
) -> list[dict]:
    """F10: downsample any repo whose share exceeds PER_REPO_CAP_FRACTION.

    Cap = ceil(PER_REPO_CAP_FRACTION × pool_size_before_cap). If the capped
    dataset is still unbalanced (rare unless one repo dominates heavily),
    the enforcement is approximate — revisit manually if it happens.
    """
    before = len(candidates)
    cap = max(1, int(PER_REPO_CAP_FRACTION * before)) if before else 0
    by_repo: dict[str, list[dict]] = {}
    for c in candidates:
        by_repo.setdefault(c["repo"], []).append(c)
    kept: list[dict] = []
    for repo, items in by_repo.items():
        if len(items) > cap:
            rng.shuffle(items)
            items = items[:cap]
        kept.extend(items)
    counts["F10_per_repo_cap"] = {
        "before": before,
        "after": len(kept),
        "dropped": before - len(kept),
        "cap_per_repo": cap,
    }
    return kept


def print_report(counts: dict[str, dict], final: list[dict]) -> None:
    print("\n=== Filter survival report ===")
    for name, info in counts.items():
        extra = f"  cap={info['cap_per_repo']}" if "cap_per_repo" in info else ""
        print(
            f"  {name:22s}  {info['before']:>6d} → {info['after']:>6d}  "
            f"(dropped {info['dropped']}){extra}"
        )
    total = len(final)
    print(f"\n  Final pairs: {total}")
    if not total:
        return
    print("\n  Per-repo distribution:")
    for repo, n in Counter(c["repo"] for c in final).most_common():
        pct = n / total * 100
        print(f"    {repo:40s} {n:>5d} ({pct:4.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-file", default=DEFAULT_OUT)
    parser.add_argument("--sample-file", default=DEFAULT_SAMPLE)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-repo-cap",
        action="store_true",
        help="disable F10 per-repo cap (use for single-repo held-out sets)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"Raw dir not found: {raw_dir}", file=sys.stderr)
        return 1

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file = Path(args.sample_file)
    sample_file.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    candidates = explode_to_candidates(load_raw(raw_dir))
    counts: dict[str, dict] = {
        "raw_candidates": {"before": 0, "after": len(candidates), "dropped": 0}
    }

    filters: list[tuple[str, Callable[[dict], bool]]] = [
        ("F7_has_file_path", lambda c: bool(c["file_path"])),
        (
            "F7b_not_self_reply",
            lambda c: c["comment_user"] is not None
            and c["comment_user"] != c["pr_author"],
        ),
        ("F1_python_only", lambda c: c["file_path"].endswith(".py")),
        ("F6_has_merge_date", lambda c: c["pr_merged_at"] is not None),
        ("F5_human_author", lambda c: not is_bot(c["comment_user"])),
        (
            "F4_comment_length",
            lambda c: MIN_COMMENT_CHARS <= len(c["comment"]) <= MAX_COMMENT_CHARS,
        ),
        (
            "F3_hunk_size",
            lambda c: MIN_HUNK_LINES
            <= len(c["diff_hunk"].splitlines())
            <= MAX_HUNK_LINES,
        ),
        ("F8_no_external_refs", lambda c: not has_external_ref(c["comment"])),
    ]
    for name, pred in filters:
        candidates = apply_filter(name, candidates, pred, counts)

    candidates = dedupe(candidates, counts)
    if args.no_repo_cap:
        counts["F10_per_repo_cap"] = {
            "before": len(candidates),
            "after": len(candidates),
            "dropped": 0,
            "cap_per_repo": "disabled",
        }
    else:
        candidates = apply_repo_cap(candidates, counts, rng)

    rng.shuffle(candidates)

    with out_file.open("w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    with sample_file.open("w") as f:
        for c in candidates[: args.sample_size]:
            f.write(json.dumps(c) + "\n")

    print_report(counts, candidates)
    print(f"\n  Output: {out_file}")
    print(f"  Sample: {sample_file}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
