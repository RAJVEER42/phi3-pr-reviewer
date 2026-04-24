# Results — Per-row Predictions

Frozen prediction files from each evaluation run. These are the actual outputs the reported metrics were computed on, committed here so anyone can audit or re-score them without re-running generation on a GPU.

Each row is JSON with at minimum: `idx`, `prediction`, `reference`, `bertscore_f1`, `bertscore_precision`, `bertscore_recall`, plus metadata (`repo`, `pr_number`, `file_path`).

## Files

| File | What it is | Rows |
|---|---|---|
| `finetuned_predictions.jsonl` | Fine-tuned Phi-3-mini (base + LoRA adapter) on the **in-domain test split** of [`Itachi-42/pr-reviews`](https://huggingface.co/datasets/Itachi-42/pr-reviews). | 584 |
| `baseline_predictions.jsonl` *(to be added)* | Zero-shot base Phi-3-mini-4k-instruct on the same test split — the comparison baseline. | 584 |
| `finetuned_holdout_predictions.jsonl` *(to be added)* | Fine-tuned model on the held-out repo [`Itachi-42/pr-reviews-holdout`](https://huggingface.co/datasets/Itachi-42/pr-reviews-holdout) (`psf/black`) — generalization check. | ~81 |

## Headline numbers

From `finetuned_predictions.jsonl`:
- **BERTScore F1:** mean 0.4667, median 0.479 (std 0.088, p10/p90 = 0.339/0.574)
- **Output length:** median 19.5 words (reference median: 18) — the fine-tune matches reference terseness on the median, but a ~10% tail of code-continuation failures inflates the mean to 38.2.

See [RESULTS.md](../RESULTS.md) for the full table and honest shortfall analysis.

## How these were produced

```bash
python scripts/run_eval.py \
    --dataset Itachi-42/pr-reviews \
    --split test \
    --adapter Itachi-42/phi3-pr-reviewer-lora \
    --tag finetuned \
    --out-dir /kaggle/working
```

Source: [scripts/run_eval.py](../scripts/run_eval.py).
Scorer: `microsoft/deberta-xlarge-mnli` (via `bert-score`), tokenizer capped at `model_max_length=512` to work around an overflow in newer `transformers`.
