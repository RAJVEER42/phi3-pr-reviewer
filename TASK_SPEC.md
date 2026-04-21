# Task Specification — Phi-3-mini PR Reviewer

**Status:** FROZEN — changing any section invalidates earlier work.
**Last updated:** 2026-04-21

This document defines exactly what the model does, what it sees, what it produces, and how we judge it. Every downstream decision (scraper filters, tokenizer max length, eval harness) derives from what's written here.

---

## 1. Problem statement

**Does:** Given a single-hunk Python code change from a GitHub pull request, generate one inline review comment that a competent human reviewer might have written.

**Does not:**
1. Generate full PR summaries or release notes.
2. Replace static analysis / linters (no AST checks, no type inference).
3. Reason across multiple hunks, files, or conversation turns.

---

## 2. Input format

The model receives a single user message using the Phi-3 chat template:

```
<|user|>
Review this code change:
```diff
<UNIFIED_DIFF_HUNK>
```<|end|>
<|assistant|>
```

### Input constraints
| Field | Rule |
|---|---|
| Language | Python only (`.py` files) |
| Diff format | Unified diff, single hunk, `±10` lines of context around the changed region |
| Max hunk length | 300 lines total (including context) |
| Max tokens (input side) | 1,800 tokens (leaves ~250 for output inside 2,048-ctx) |
| File count | Exactly 1 file per example |
| Hunk count | Exactly 1 hunk per example |

No filename, no PR title, no PR description, no commit message, no surrounding code is included. The model sees the diff and nothing else. Deliberate choice: forces the model to reason from code, not from prose cues.

---

## 3. Output format

A single inline review comment as plain text. No markdown headers, no bullet lists longer than 3 items, no code blocks unless quoting a specific identifier or one-line suggestion.

### Output constraints
| Field | Rule |
|---|---|
| Length | 20–200 tokens (hard cap at 256 via `max_new_tokens`) |
| Style | Second-person or imperative ("Consider…", "This will raise when…"), matching OSS review tone |
| Scope | One issue per comment — the most important one. Not a laundry list. |
| Allowed topics | Correctness bugs, API misuse, perf issues, readability, idiom, test coverage gaps, error-handling gaps |
| Forbidden | Praise-only ("LGTM", "nice"), meta-comments ("see PR #1234"), questions with no actionable suggestion, emoji |

---

## 4. Scope limits (what counts as "in distribution")

Only pairs meeting **all** of these enter training, validation, or test:

1. **Language:** Python (`.py`). No `.pyi`, `.ipynb`, `.pyx`, config, docs, or YAML.
2. **File count:** PR touched 1+ files, but each training row uses exactly 1 file.
3. **Hunk size:** 5 ≤ total hunk lines ≤ 300 (including context).
4. **Comment length:** 20 ≤ chars ≤ 500.
5. **Comment author:** Human (not a bot). Bot blocklist: `dependabot[bot]`, `pre-commit-ci[bot]`, `codecov[bot]`, `codecov-io[bot]`, `github-actions[bot]`, `renovate[bot]`, `mergify[bot]`, `sonarcloud[bot]`, `allcontributors[bot]`. Extend as new bots appear.
6. **PR status:** Closed AND merged. Rejected PRs' reviews are noisy (style disagreements, closed-as-wontfix).
7. **Comment type:** Inline review comment tied to a specific file + line. General PR conversation comments are excluded.
8. **No external references:** Drop comments containing `#\d+` issue links, "as discussed", "per the RFC", "see the doc", or "offline".
9. **No duplicates:** Same (normalized-comment, same-reviewer) across PRs → keep one.
10. **Repo diversity:** No single repo may contribute more than 15% of final pairs (avoids domain collapse).

---

## 5. Dataset splits

Total target: **4,000 pairs** after filtering.

| Split | Size | Purpose |
|---|---|---|
| train | 3,600 (90%) | Gradient updates |
| validation | 200 (5%) | Early stopping, LR tuning during training |
| test | 200 (5%) | Held-out final eval — **touched only once, at the end** |

**Stratification:** By source repo. Every repo present in train must also appear in val and test (proportional to its share).

**Held-out-repo set:** Additionally scrape ~100 pairs from `psf/black` — a repo **deliberately excluded** from the 30-repo source list. Used exclusively for a generalization sanity check in Phase 5. If `psf/black` yields too few pairs after filtering, fall back to `python-poetry/poetry`.

---

## 6. Evaluation metrics

### 6.1 Primary (automated)
**BERTScore F1** (`microsoft/deberta-xlarge-mnli` as the scorer model) between generated comment and the reference reviewer comment, averaged over the 200-example test set.

Reported:
- Baseline (zero-shot Phi-3-mini) BERTScore
- Fine-tuned BERTScore
- Absolute delta + % improvement

**Target:** ≥10% relative improvement over baseline. Below that → project is not shippable, go back and investigate.

### 6.2 Secondary (manual)
**50-example rubric.** Random 50 from the test set, scored 1–5 on each of:

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Relevance | Off-topic or hallucinated | Touches the right area, misses specific issue | Identifies the real issue the human flagged |
| Actionability | No concrete suggestion | Vague ("consider refactoring") | Specific, implementable change |
| Factuality | Wrong claim about the code | Ambiguous | Verifiable from the diff alone |

Reported as mean scores per dimension, baseline vs. fine-tuned.

### 6.3 Tertiary (qualitative)
- 5 wins (fine-tuned clearly better than baseline) — include in README.
- 5 losses (baseline better or both bad) — include in README.
- Held-out-repo BERTScore — catches memorization.

---

## 7. Success criteria

Four thresholds. Each has a pre-committed response plan so a failure triggers a specific action, not aimless retraining.

| # | Threshold | If it fails → do this |
|---|---|---|
| **S1** | Fine-tuned BERTScore ≥ baseline + **10% relative** | Stop. Re-examine dataset quality (filter counts, random sample review) before retraining. |
| **S2** | Held-out-repo BERTScore within **5%** of in-domain test BERTScore | Cut epochs to 1, add `lora_dropout=0.05`, retrain. |
| **S3** | Manual rubric **relevance ≥ 3.5/5** mean on fine-tuned (and meaningfully above baseline) | Dataset issue, not training issue. Revisit filter F3 (hunk size) and F4 (comment length). |
| **S4** | Output-format compliance (per §3) **≥ 95%** | Audit training-data formatting for drift (verify literal chat-template bytes). Retrain. |

If any threshold fails → **do not ship.** Diagnose, execute the response plan, re-evaluate.

---

## 8. Explicit non-goals

- Catching real bugs in a statistically meaningful way (we're cloning reviewer *style*, not replacing review).
- Beating GPT-4 or Claude on this task. Goal is demonstrable improvement of a small model on a narrow task.
- Multi-language support.
- Multi-hunk / whole-PR reasoning.
- RLHF / DPO. SFT only.
