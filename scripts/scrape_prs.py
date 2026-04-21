#!/usr/bin/env python3
"""Phase 2 scraper — pull (diff, review_comment) raw pairs from GitHub.

Reads repo slugs from config/sources.txt, fetches merged PRs in the last
N years, and writes one JSONL per repo to data/raw/.

Output row = one merged PR with >=1 inline review comment. Later phases
filter these into (hunk, comment) training pairs.

Resumable: re-runs skip PRs already written to the per-repo JSONL.
Rate-limit aware: sleeps until reset when remaining < 50.

Usage:
    # Smoke test (3 PRs from one repo):
    python scripts/scrape_prs.py --repos pallets/click --limit-per-repo 3

    # Full run (all 30 repos, last 2 years):
    python scripts/scrape_prs.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from github import Auth, Github, GithubException, RateLimitExceededException
from tqdm import tqdm

DEFAULT_SOURCES = "config/sources.txt"
DEFAULT_OUT = "data/raw"
DEFAULT_YEARS_BACK = 2
RATE_LIMIT_BUFFER = 50  # sleep when remaining drops below this

logger = logging.getLogger("scrape_prs")


def load_repos(sources_path: Path) -> list[str]:
    repos: list[str] = []
    for line in sources_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            repos.append(line)
    return repos


def already_scraped(out_file: Path) -> set[int]:
    if not out_file.exists():
        return set()
    seen: set[int] = set()
    with out_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["pr_number"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def wait_for_rate_limit(g: Github) -> None:
    remaining, _limit = g.rate_limiting
    if remaining >= RATE_LIMIT_BUFFER:
        return
    reset_epoch = g.rate_limiting_resettime
    wait_s = max(0.0, reset_epoch - time.time() + 5)
    logger.warning(
        "Rate limit low (%d remaining). Sleeping %.0fs until reset.",
        remaining,
        wait_s,
    )
    time.sleep(wait_s)


def serialize_pr(pr) -> dict:
    files = [
        {"filename": f.filename, "patch": f.patch or ""}
        for f in pr.get_files()
    ]
    comments = []
    for c in pr.get_review_comments():
        comments.append(
            {
                "id": c.id,
                "user": c.user.login if c.user else None,
                "body": c.body,
                "path": c.path,
                "position": c.position,
                "original_position": c.original_position,
                "commit_id": c.commit_id,
                "diff_hunk": c.diff_hunk,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
        )
    return {
        "repo": pr.base.repo.full_name,
        "pr_number": pr.number,
        "pr_title": pr.title,
        "pr_body": pr.body or "",
        "author": pr.user.login if pr.user else None,
        "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
        "files": files,
        "review_comments": comments,
    }


def in_window(dt, since, until) -> bool:
    if dt is None:
        return False
    dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return since <= dt <= until


def scrape_repo(
    g: Github,
    slug: str,
    out_dir: Path,
    since: datetime,
    until: datetime,
    limit: int | None,
) -> int:
    out_file = out_dir / f"{slug.replace('/', '__')}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    seen = already_scraped(out_file)
    logger.info("[%s] %d PRs already scraped (resuming)", slug, len(seen))

    try:
        repo = g.get_repo(slug)
    except GithubException as e:
        logger.error("[%s] cannot access repo: %s", slug, e)
        return 0

    pulls = repo.get_pulls(state="closed", sort="updated", direction="desc")

    written = 0
    pbar = tqdm(desc=slug, unit="pr", leave=True)
    with out_file.open("a") as f:
        for pr in pulls:
            pbar.update(1)

            # Early stop: pulls sorted by updated_at desc, so once we see a PR
            # updated before our window, everything after is too old.
            if pr.updated_at and not in_window(pr.updated_at, since, until):
                if pr.updated_at.replace(tzinfo=timezone.utc) < since:
                    break

            if pr.number in seen:
                continue
            if pr.merged_at is None:
                continue
            if not in_window(pr.merged_at, since, until):
                continue

            wait_for_rate_limit(g)
            try:
                row = serialize_pr(pr)
            except RateLimitExceededException:
                wait_for_rate_limit(g)
                try:
                    row = serialize_pr(pr)
                except GithubException as e:
                    logger.warning("[%s] PR #%d failed after retry: %s", slug, pr.number, e)
                    continue
            except GithubException as e:
                logger.warning("[%s] PR #%d failed: %s", slug, pr.number, e)
                continue

            # No inline review comments → no training signal. Skip.
            if not row["review_comments"]:
                continue

            f.write(json.dumps(row) + "\n")
            f.flush()
            written += 1
            pbar.set_postfix(kept=written)

            if limit is not None and written >= limit:
                break
    pbar.close()
    logger.info("[%s] wrote %d new rows → %s", slug, written, out_file)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument(
        "--repos",
        help="comma-separated repo slugs, overrides --sources",
    )
    parser.add_argument(
        "--limit-per-repo",
        type=int,
        default=None,
        help="cap on kept PRs per repo (smoke tests)",
    )
    parser.add_argument("--years-back", type=int, default=DEFAULT_YEARS_BACK)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN missing — put it in .env", file=sys.stderr)
        return 1

    g = Github(auth=Auth.Token(token), per_page=100)

    if args.repos:
        repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    else:
        repos = load_repos(Path(args.sources))

    until = datetime.now(timezone.utc)
    since = until - timedelta(days=365 * args.years_back)
    out_dir = Path(args.out_dir)

    logger.info(
        "Scraping %d repos, merged-window %s → %s",
        len(repos),
        since.date(),
        until.date(),
    )

    total = 0
    for slug in repos:
        try:
            total += scrape_repo(g, slug, out_dir, since, until, args.limit_per_repo)
        except KeyboardInterrupt:
            logger.warning("Interrupted. Partial output preserved.")
            break
        except Exception as e:
            logger.error("[%s] unexpected failure: %s", slug, e)
            continue

    logger.info("Done. Total new rows across repos: %d", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
