# Results

Locked-in metrics from each phase, for comparison and README.

## Phase 3 — Baseline (zero-shot Phi-3-mini-4k-instruct)

| Metric | Value |
|---|---|
| **BERTScore F1** | **0.4316** |
| BERTScore Precision | 0.3781 |
| BERTScore Recall | 0.5102 |
| n_examples | 584 (test split) |
| dataset | `Itachi-42/pr-reviews` |
| model | `microsoft/Phi-3-mini-4k-instruct` (fp16, greedy decode, max_new_tokens=256) |
| hardware | Kaggle T4 |
| date | 2026-04-22 |
| run | https://wandb.ai/irajveer-bishnoi2310-bits-pilani42/phi3-pr-reviewer/runs/unpjr1t0 |

**Observation:** recall > precision by 13 points — zero-shot model is verbose, producing filler alongside real content. Fine-tuning should narrow outputs toward the terse OSS-reviewer style, raising precision without losing recall.

## Phase 4 — Fine-tuned (QLoRA)

_To be filled in after Phase 4 run._

## Phase 5 — Held-out generalization (`psf/black`)

_To be filled in after Phase 5 run._

---

## Success thresholds (from TASK_SPEC §7)

| # | Threshold | Baseline | Status |
|---|---|---|---|
| S1 | Fine-tuned F1 ≥ baseline + 10% relative (≥ 0.4748) | 0.4316 | pending |
| S2 | Held-out F1 within 5% of in-domain F1 | — | pending |
| S3 | Manual rubric relevance ≥ 3.5/5 | — | pending |
| S4 | Output-format compliance ≥ 95% | — | pending |
