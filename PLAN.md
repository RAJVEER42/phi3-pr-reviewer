# Fine-Tuning Phi-3-mini on GitHub PRs — Project Plan

End-to-end roadmap: from zero to a shipped, measurable fine-tuned code-review model.

**Target model:** `microsoft/Phi-3-mini-4k-instruct` (3.8B params)
**Training venue:** Kaggle P100 (free, 30 hrs/week)
**Stack:** Unsloth + QLoRA + TRL's SFTTrainer + Hugging Face Datasets + Weights & Biases

---

## Phase 0 — Environment & Accounts (30 min)

### Accounts (all free)
1. **Hugging Face** (`huggingface.co`) — access token with `write` scope. Hosts dataset and model.
2. **Kaggle** (`kaggle.com`) — phone-verified for free P100 GPU (30 hrs/week). Primary training venue.
3. **Weights & Biases** (`wandb.ai`) — free tier, unlimited personal projects. Experiment tracking.
4. **GitHub** — personal access token with `repo` scope (read-only works). Feeds the PR scraper.

### Local setup (Mac)
```bash
cd /Users/rajveerbishnoi/Project_fine_tune
python3 -m venv .venv
source .venv/bin/activate
pip install datasets transformers huggingface_hub PyGithub pandas python-dotenv tqdm
```

### Secrets
Create `.env` (gitignored, never committed):
```
GITHUB_TOKEN=ghp_xxx
HF_TOKEN=hf_xxx
WANDB_API_KEY=xxx
```

---

## Phase 1 — Define the Task Precisely (1 hr, non-skippable)

Lock these down **before** writing any code. Changing them mid-project is the #1 time sink.

1. **Input format** — exact string the model sees.
2. **Output format** — exact string the model produces (length, style, scope).
3. **Scope limits** — language, file count, diff size, PR type.
4. **Evaluation metric** — primary automated metric + manual rubric.

Output: [TASK_SPEC.md](TASK_SPEC.md) — the frozen spec for the rest of the project.

---

## Phase 2 — Build the Dataset (4–7 days — the actual project)

**Target:** 4,000 clean `(diff_hunk, review_comment)` pairs. Scrape ~8,000 raw, filter to 4,000.

### 2.1 Source list (half a day)
Pick ~30 active Python OSS repos with rigorous review culture: `pandas-dev/pandas`, `scikit-learn/scikit-learn`, `pallets/flask`, `django/django`, `tiangolo/fastapi`, etc. Diversity → robustness.

### 2.2 Scrape PRs (1 day) — `scripts/scrape_prs.py`
For each repo, pull closed+merged PRs from the last ~2 years. Collect:
- Unified diff (via `PullRequest.get_files()` patches)
- Review comments with file + line context (via `PullRequest.get_review_comments()`)
- PR title, description, merged_at, repo, PR number
Respect GitHub rate limits (5000 req/hr with token). Save raw JSONL per repo.

### 2.3 Filter aggressively (1 day) — **where the craft lives**
Drop:
- Diffs >300 lines
- Non-Python files
- Comments <20 chars ("lgtm", "nit") or >500 chars (essays)
- Pure-praise / emoji-only comments
- Bot comments (Dependabot, pre-commit.ci, CodeCov, etc.)
- Comments referencing external context ("as discussed in #1234", "per the RFC")
- Duplicates

**Document every filter with counts.** "47,320 raw → 29,216 after length → 23,016 after bot filter → 4,016 final." This table goes in the README — interviewers ask about this.

### 2.4 Pair diffs with comments (half a day)
Each review comment has a file + line range. Extract the diff hunk ±10 lines around it. One `(hunk, comment)` pair per row — focused signal beats "here's 200 lines, guess which one I meant."

### 2.5 Format as instruction pairs (2 hrs)
Phi-3 chat template:
```
<|user|>
Review this code change:
```diff
<hunk>
```<|end|>
<|assistant|>
<review_comment><|end|>
```
Split 90/5/5 → train/val/test. **Stratify by repo** so every repo appears in every split.

### 2.6 Push to HF Hub (15 min)
```python
from datasets import DatasetDict
ds = DatasetDict({"train": ..., "validation": ..., "test": ...})
ds.push_to_hub("your-username/pr-reviews", private=True)
```

---

## Phase 3 — Baseline First (2 hrs, non-skippable)

On a Kaggle P100 notebook:
1. Load `microsoft/Phi-3-mini-4k-instruct` in 4-bit (`bitsandbytes`).
2. Run zero-shot inference on all 200 test-set examples with a well-crafted prompt.
3. Compute BERTScore vs. reference comments. Log to W&B.
4. **Save this number.** It's the bar you have to beat.

Sanity checks:
- Baseline already great → task too easy, narrow scope.
- Baseline incoherent → prompt/format bug, fix before training.

---

## Phase 4 — Fine-Tune with QLoRA (1 day)

### 4.1 Kaggle notebook deps
```python
!pip install "unsloth[kaggle] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl peft accelerate bitsandbytes wandb
```

### 4.2 Load in 4-bit via Unsloth
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3-mini-4k-instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
```

### 4.3 Starting hyperparameters
| Hyperparameter | Value |
|---|---|
| learning_rate | 2e-4 |
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 4 (effective batch 8) |
| num_train_epochs | 2 (LoRA overfits fast at 3+) |
| warmup_ratio | 0.03 |
| lr_scheduler_type | cosine |
| optim | adamw_8bit |
| max_seq_length | 2048 |
| report_to | wandb |

Expect 3–4 hrs on P100.

### 4.4 Training-loss diagnostics
- **Train loss drops, val loss rises after epoch 1** → overfit. Cut to 1 epoch or lower LR.
- **Train loss flat** → LR too low. Try 3e-4.
- **Train loss NaN/spiky** → LR too high OR bad tokenization. Inspect a few batches.

---

## Phase 5 — Evaluate Honestly (half day)

1. Run fine-tuned model on the **exact same** 200 test examples from Phase 3.
2. Compute BERTScore. Report baseline and fine-tuned side-by-side — not just yours.
3. Manually score 50 examples, 1–5, on: relevance, actionability, factuality.
4. Find 5 wins and 5 losses vs. baseline. Put them in the README. Honest failure modes > cherry-picked wins.
5. Held-out-repo test: run on a repo the model never saw during training. Catches memorization.

---

## Phase 6 — Ship It (half day)

1. Push LoRA adapter to HF Hub: `model.push_to_hub("your-username/phi3-pr-reviewer-lora")`. ~100 MB.
2. Build a Gradio demo — paste a diff, get a review. Host on HF Spaces (free).
3. README content:
   - Problem statement
   - Data pipeline with filter-count table
   - Baseline vs. fine-tuned BERTScore
   - Example outputs (wins + losses)
   - Failure modes
   - "What I'd do next"
4. Short blog post / LinkedIn writeup linking the Space, the model, and the repo.

---

## What this signals to a hiring manager

- You understand the full pipeline: **data → train → eval → deploy**.
- You can measure your own work honestly (baseline + held-out repo + failure modes).
- You shipped something real.

More valuable than knowing what LoRA stands for.

---

## Non-negotiable principles

1. **The dataset is the project.** Fine-tuning takes hours. Building 4,000 clean pairs from real PRs takes a week.
2. **The baseline kills most projects.** No baseline number = no project, just a notebook.
3. **Smallest model that makes sense.** Phi-3-mini before Llama 3 8B before anything larger.
4. **Narrow scope wins.** A model that reviews Python unit-test PRs well beats one that reviews anything poorly.
