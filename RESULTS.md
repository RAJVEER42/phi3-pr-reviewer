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

**Observation:** recall > precision by 13 points — zero-shot model is verbose, producing filler alongside real content. Fine-tuning should narrow outputs toward terse OSS-reviewer style, raising precision without losing recall.

## Phase 4 — Fine-tuned (QLoRA adapter on Phi-3-mini)

| Metric | Baseline | Fine-tuned | Δ (absolute) | Δ (relative) |
|---|---|---|---|---|
| **BERTScore F1** | 0.4316 | **0.4667** | +0.0351 | **+8.13%** |
| BERTScore Precision | 0.3781 | **0.4807** | +0.1026 | **+27.1%** |
| BERTScore Recall | 0.5102 | 0.4730 | −0.0372 | −7.3% |

Metadata: n=584 test examples; adapter = `Itachi-42/phi3-pr-reviewer-lora`; training = 2,000 / 2,512 planned steps (~80% of 2 epochs); eval loss 2.464 → 2.362; hardware Kaggle T4×2 via Unsloth; date 2026-04-23.

**Observation:** the fine-tune did exactly what the verbosity failure mode of the baseline predicted. Precision jumped +27% because the model learned to stop rambling; recall dropped modestly because terser outputs inevitably miss some reference content. Net F1 lift was +8.13% relative — meaningful but short of the pre-committed 10% threshold. Training was cut short at ~80% of plan due to a Kaggle eval-hang; the adapter was recovered from checkpoint-2000.

**Prediction-level analysis** (see [results/README.md](results/README.md) for details):
- F1 **median = 0.479**, mean = 0.467 (std 0.088). Median > mean indicates a left-skewed tail dragging down the average.
- Output-length **median = 19.5 words vs. reference median = 18** — fine-tune matched reference terseness on the median.
- But output-length **mean = 38.2 vs. reference mean = 22.8** — ~10% of outputs fall into a *"continue the diff"* failure mode (model regenerates code instead of producing a review comment), which tanks F1 on those rows from ~0.5 to ~0.25 and pulls the mean down.

## Phase 5 — Held-out generalization (`psf/black`)

_To be filled in after Phase 5 run (81 pairs from `psf/black`, excluded from training)._

---

## Success thresholds (from TASK_SPEC §7)

| # | Threshold | Observed | Status |
|---|---|---|---|
| S1 | Fine-tuned F1 ≥ baseline + 10% relative (≥ 0.4748) | 0.4667 (+8.13% rel) | ❌ **missed by ~2 pp** |
| S2 | Held-out F1 within 5% of in-domain F1 | — | pending |
| S3 | Manual rubric relevance ≥ 3.5/5 | — | pending |
| S4 | Output-format compliance ≥ 95% | — | pending |

## Honest reading

The fine-tune meaningfully improved the model (+8.13% relative F1 = +3.5 absolute points), but **fell short of the pre-committed 10% shippability threshold by ~2 percentage points.** Most probable cause: training was cut at step 2000 of 2512 (~80%), so the final ~500 steps — where the cosine schedule is lowest and typically delivers the last few points of metric gain — did not run.

Per TASK_SPEC §7 S1's pre-written response plan (*"if fine-tuned F1 < baseline + 10%, stop and re-examine dataset quality before retraining"*), the project is **not yet shippable on the primary metric**, pending diagnosis.

Next steps:
1. Held-out (`psf/black`) eval — if generalization is clean, the 8.13% is a defensible result even below threshold.
2. Only then decide whether to retrain to completion.

Reporting 8.13% honestly (rather than chasing the 10% number through post-hoc threshold manipulation) is the intended behavior of the pre-committed-thresholds discipline.
