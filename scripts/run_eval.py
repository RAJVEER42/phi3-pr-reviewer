#!/usr/bin/env python3
"""Phase 3 / Phase 5 — evaluate Phi-3-mini (base or base+LoRA) on the test set.

Same script used for both:
    Phase 3: baseline (no adapter) — establishes the bar the fine-tune must beat.
    Phase 5: fine-tuned (--adapter path_or_hub_id) — measured on the identical
             test set so the delta is honest.

Pipeline:
    1. Load Phi-3-mini in 4-bit. Optionally attach a LoRA adapter.
    2. Load test split from HF Hub.
    3. Greedy-decode a review comment per row (deterministic → reproducible delta).
    4. Compute BERTScore F1 (deberta-xlarge-mnli scorer) per TASK_SPEC §6.1.
    5. Log mean metrics to W&B. Save per-row predictions as JSONL.

Designed to run on a single Kaggle P100 (free tier). Secrets come from env
vars locally or Kaggle Secrets on the notebook. See KAGGLE_SETUP.md.

Usage:
    # Baseline (Phase 3):
    python scripts/run_eval.py --dataset your-username/pr-reviews --tag baseline

    # Fine-tuned (Phase 5):
    python scripts/run_eval.py \\
        --dataset your-username/pr-reviews \\
        --adapter your-username/phi3-pr-reviewer-lora \\
        --tag finetuned

    # Held-out repo (Phase 5, generalization check):
    python scripts/run_eval.py \\
        --dataset your-username/pr-reviews-holdout \\
        --adapter your-username/phi3-pr-reviewer-lora \\
        --tag finetuned_holdout
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
SCORER_MODEL = "microsoft/deberta-xlarge-mnli"
MAX_NEW_TOKENS = 256  # TASK_SPEC §3 hard cap


def get_secret(name: str) -> str | None:
    """Read a secret from env or Kaggle Secrets, in that order."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def load_base_model(use_4bit: bool = False):
    """Load Phi-3-mini.

    For eval we default to fp16 (no quantization) — Phi-3-mini is 3.8B params
    ≈ 7.6 GB in fp16, fits easily in a T4. 4-bit dequantization overhead on T4
    is severe (~5x slower than fp16 because sm_75 lacks modern dequant paths).
    The LoRA adapter from Phase 4 still loads fine on top of an fp16 base via
    PeftModel.from_pretrained.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # T4 has no native bf16; auto-pick the best supported dtype.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    kwargs = {"device_map": "auto", "torch_dtype": compute_dtype}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def attach_adapter(model, adapter_ref: str, hf_token: str | None = None):
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_ref, token=hf_token)
    model.eval()
    return model


def generate_batched(
    model,
    tokenizer,
    user_contents: list[str],
    batch_size: int = 4,
) -> list[str]:
    """Batched greedy decoding. Left-padding is required for causal-LM batched generate.

    Yields the generated text for each input, in input order. Keeps greedy
    (do_sample=False) so each example's output is identical to the single-example
    path — batching changes wall time, not results.
    """
    import torch
    from tqdm import tqdm

    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        results: list[str] = []
        for i in tqdm(range(0, len(user_contents), batch_size), desc="generating"):
            chunk = user_contents[i : i + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": c}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for c in chunk
            ]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1792,  # leaves ~256 tokens headroom for generation
            ).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_len = inputs.input_ids.shape[1]
            for seq in out:
                text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                results.append(text)
        return results
    finally:
        tokenizer.padding_side = prev_padding_side


def run(args) -> int:
    import wandb
    from datasets import load_dataset
    from tqdm import tqdm

    hf_token = get_secret("HF_TOKEN")
    wb_key = get_secret("WANDB_API_KEY")
    if wb_key:
        os.environ["WANDB_API_KEY"] = wb_key

    wandb.init(
        project=args.wandb_project,
        name=args.tag,
        config={
            "base_model": BASE_MODEL,
            "adapter": args.adapter or None,
            "dataset": args.dataset,
            "split": args.split,
            "scorer": SCORER_MODEL,
        },
    )

    precision_tag = "4-bit" if args.use_4bit else "fp16/bf16"
    print(f"Loading base model {BASE_MODEL} ({precision_tag})...")
    model, tokenizer = load_base_model(use_4bit=args.use_4bit)

    if args.adapter:
        print(f"Attaching adapter {args.adapter}...")
        model = attach_adapter(model, args.adapter, hf_token=hf_token)
    model.eval()

    print(f"Loading dataset {args.dataset} / {args.split}...")
    ds = load_dataset(args.dataset, split=args.split, token=hf_token)
    print(f"  {len(ds)} rows")

    user_contents: list[str] = []
    references: list[str] = []
    metadata: list[dict] = []
    for row in ds:
        user_msg = next(m for m in row["messages"] if m["role"] == "user")
        ref = next(m for m in row["messages"] if m["role"] == "assistant")
        user_contents.append(user_msg["content"])
        references.append(ref["content"])
        metadata.append(
            {
                "repo": row.get("repo"),
                "pr_number": row.get("pr_number"),
                "file_path": row.get("file_path"),
            }
        )

    predictions = generate_batched(
        model, tokenizer, user_contents, batch_size=args.batch_size
    )

    print("\nComputing BERTScore (this loads ~1.5 GB scorer on first run)...")
    from bert_score import score as bert_score

    P, R, F = bert_score(
        predictions,
        references,
        model_type=SCORER_MODEL,
        lang="en",
        batch_size=8,
        verbose=False,
    )
    mean_f = F.mean().item()
    mean_p = P.mean().item()
    mean_r = R.mean().item()

    wandb.log(
        {
            "bertscore_f1": mean_f,
            "bertscore_precision": mean_p,
            "bertscore_recall": mean_r,
            "n_examples": len(predictions),
        }
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"{args.tag}_predictions.jsonl"
    summary_path = out_dir / f"{args.tag}_summary.json"

    with preds_path.open("w") as f:
        for i, (p, r, m) in enumerate(zip(predictions, references, metadata)):
            f.write(
                json.dumps(
                    {
                        "idx": i,
                        "prediction": p,
                        "reference": r,
                        "bertscore_f1": F[i].item(),
                        "bertscore_precision": P[i].item(),
                        "bertscore_recall": R[i].item(),
                        **m,
                    }
                )
                + "\n"
            )

    with summary_path.open("w") as f:
        json.dump(
            {
                "tag": args.tag,
                "base_model": BASE_MODEL,
                "adapter": args.adapter,
                "dataset": args.dataset,
                "split": args.split,
                "n_examples": len(predictions),
                "bertscore_f1": mean_f,
                "bertscore_precision": mean_p,
                "bertscore_recall": mean_r,
            },
            f,
            indent=2,
        )

    print(f"\n  BERTScore F1: {mean_f:.4f}")
    print(f"  Predictions: {preds_path}")
    print(f"  Summary:     {summary_path}")
    wandb.finish()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="HF dataset repo id")
    parser.add_argument("--split", default="test")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path or hub id (optional)")
    parser.add_argument("--tag", required=True, help="run name, used for output filenames")
    parser.add_argument("--wandb-project", default="phi3-pr-reviewer")
    parser.add_argument("--out-dir", default="/kaggle/working")
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="load base in 4-bit (slower on T4; only use if VRAM-constrained)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="generation batch size (T4 handles 4 safely; T4x2/A100 can go higher)",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
