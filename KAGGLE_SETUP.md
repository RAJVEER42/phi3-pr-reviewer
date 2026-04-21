# Kaggle Setup — Running Phases 3, 4, 5 on a Free P100 GPU

Everything in Phases 3–5 runs on a single Kaggle P100 notebook. Free tier: 30 GPU-hrs / week. The three scripts are independent — you can close and reopen between phases.

---

## One-time account setup

1. **Sign in at https://www.kaggle.com/** and **verify your phone** under Settings → Account. Phone verification unlocks free GPU access.
2. **Add two secrets** at https://www.kaggle.com/settings (scroll to Secrets):
   - `HF_TOKEN` — your Hugging Face write token.
   - `WANDB_API_KEY` — your Weights & Biases key (from https://wandb.ai/authorize).
   These are exposed to notebooks via `UserSecretsClient`; the scripts already handle this.

---

## Per-run notebook setup

Always:
1. **New notebook** → Click "Create" → "New Notebook".
2. **Attach GPU** in the right-hand sidebar → *Accelerator* → **GPU P100** (not T4 — T4 is slower for bfloat16 ops).
3. **Attach the secrets**: right sidebar → *Add-ons* → *Secrets* → toggle on `HF_TOKEN` and `WANDB_API_KEY`.
4. **Internet on**: right sidebar → *Internet* → **On**. Needed to pull models and datasets.
5. **Upload this repo** to the Kaggle environment. Two options:
   - **Easy**: paste the whole repo as a single Kaggle Dataset (upload folder), then use it via `/kaggle/input/…`.
   - **Better**: clone from GitHub in cell 1: `!git clone https://github.com/your-handle/Project_fine_tune.git`.

Once set up, the three phases are copy-paste-ready notebook cells.

---

## Phase 3 — Baseline eval (~30 min)

**First cell (install):**
```python
!pip install -q transformers==4.45.0 accelerate bitsandbytes peft datasets bert-score wandb
```

**Second cell (run):**
```python
%cd /kaggle/working/Project_fine_tune   # or your path
!python scripts/run_eval.py \
    --dataset YOUR_HF_USERNAME/pr-reviews \
    --split test \
    --tag baseline \
    --out-dir /kaggle/working
```

Outputs (in `/kaggle/working/`):
- `baseline_predictions.jsonl` — per-row prediction, reference, BERTScore.
- `baseline_summary.json` — mean BERTScore F1/P/R + metadata.

**Save this number.** It's the bar Phase 4 has to beat. Per TASK_SPEC §7 S1, the fine-tune needs ≥10% relative improvement.

---

## Phase 4 — QLoRA fine-tune (~3–4 hrs)

**First cell (install Unsloth + TRL + deps):**
```python
!pip install -q "unsloth[kaggle] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q trl peft accelerate bitsandbytes datasets wandb
```

**Second cell (train):**
```python
%cd /kaggle/working/Project_fine_tune
!python scripts/finetune.py \
    --dataset YOUR_HF_USERNAME/pr-reviews \
    --hub-repo-id YOUR_HF_USERNAME/phi3-pr-reviewer-lora \
    --output-dir /kaggle/working/phi3-pr-reviewer-lora
```

Defaults come from PLAN.md §4.3: lr=2e-4, batch=2 × grad_accum=4, 2 epochs, cosine schedule, LoRA r=16. Override with CLI flags only after watching the loss curves — see PLAN.md §4.4 for what to tweak when.

**Keep the Kaggle tab open.** Kaggle auto-stops idle GPU sessions after ~1 hr of no notebook activity — a running cell counts as activity, but browser-tab suspension can still kill it on some platforms. Safer to run the `caffeinate` equivalent in your OS and leave the tab focused.

The adapter (~100 MB) pushes to HF Hub at the end. You can download it from there in Phase 5 instead of relying on Kaggle's `/kaggle/working/` persistence.

---

## Phase 5 — Fine-tuned eval + held-out (~45 min)

### 5a. Fine-tuned eval on the in-domain test split

**New notebook** (or reuse Phase 3's):
```python
!pip install -q transformers==4.45.0 accelerate bitsandbytes peft datasets bert-score wandb
```
```python
%cd /kaggle/working/Project_fine_tune
!python scripts/run_eval.py \
    --dataset YOUR_HF_USERNAME/pr-reviews \
    --split test \
    --adapter YOUR_HF_USERNAME/phi3-pr-reviewer-lora \
    --tag finetuned \
    --out-dir /kaggle/working
```

Compare `finetuned_summary.json` to `baseline_summary.json`. Your primary number is the relative BERTScore F1 delta.

### 5b. Held-out-repo generalization check

Requires the `psf/black` held-out dataset (scrape separately after Phase 2 — see project todo list).
```python
!python scripts/run_eval.py \
    --dataset YOUR_HF_USERNAME/pr-reviews-holdout \
    --split test \
    --adapter YOUR_HF_USERNAME/phi3-pr-reviewer-lora \
    --tag finetuned_holdout \
    --out-dir /kaggle/working
```

Per TASK_SPEC §7 S2: held-out BERTScore must be within 5% of in-domain BERTScore. Wider gap = overfit to training repos.

### 5c. Manual rubric (50 examples, no GPU needed)

Download `finetuned_predictions.jsonl` and `baseline_predictions.jsonl` from `/kaggle/working/`. Pick 50 random indices, score each side-by-side on the 1/3/5 rubric (TASK_SPEC §6.2). Do it in a spreadsheet. Takes ~90 min but the signal is what makes the final report credible.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` during fine-tune | Lower `--batch-size` to 1 and `--grad-accum` to 8 (effective batch unchanged). |
| Loss NaN at step 0 | Tokenizer mismatch. Verify dataset was built with the same Phi-3 tokenizer as the model. |
| Loss flat after 50 steps | Bump `--lr` to 3e-4. |
| BERTScore download fails | First run downloads ~1.5 GB for `deberta-xlarge-mnli`. Re-run; Kaggle caches it after the first successful pull. |
| Eval script can't find the adapter | After push, the repo may be private. Confirm `HF_TOKEN` secret is attached to this notebook and has Read access. |
| Session killed after 12 hrs | Kaggle hard cap. Save adapter to HF Hub mid-training via `--save-steps 100` (already default). Resume not implemented — re-running starts from scratch. |
