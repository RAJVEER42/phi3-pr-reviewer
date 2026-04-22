#!/usr/bin/env python3
"""Phase 4 — Unsloth QLoRA fine-tune of Phi-3-mini on the PR-review dataset.

Runs on a single Kaggle P100 in ~3–4 hours at default hyperparameters.

Hyperparameters match PLAN.md §4.3:
    lr=2e-4, batch=2, grad_accum=4 (effective 8), epochs=2, cosine schedule,
    warmup_ratio=0.03, adamw_8bit, max_seq_length=2048, LoRA r=16 alpha=16
    dropout=0 across all attention + MLP projection modules.

Watch these signals during training (PLAN.md §4.4):
    • train loss drops but val loss rises after epoch 1  → overfit (cut to 1 ep)
    • train loss flat                                    → lr too low (try 3e-4)
    • train loss NaN/spiky                               → lr too high OR bad tokenization

Usage (inside a Kaggle P100 notebook with this repo uploaded):
    !pip install "unsloth[kaggle] @ git+https://github.com/unslothai/unsloth.git"
    !pip install trl peft accelerate bitsandbytes wandb
    !python scripts/finetune.py \\
        --dataset your-username/pr-reviews \\
        --hub-repo-id your-username/phi3-pr-reviewer-lora
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_MODEL = "unsloth/Phi-3-mini-4k-instruct"  # identical weights to microsoft/ but Unsloth-packed
MAX_SEQ_LENGTH = 2048  # TASK_SPEC §2


def get_secret(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="HF dataset repo id")
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/phi3-pr-reviewer-lora",
        help="local dir for the trained adapter + trainer state",
    )
    parser.add_argument(
        "--hub-repo-id",
        default=None,
        help="push adapter here when training ends (optional)",
    )
    parser.add_argument("--wandb-project", default="phi3-pr-reviewer")
    parser.add_argument("--run-name", default="finetune-qlora-phi3-mini")

    # Hyperparameters (defaults per PLAN.md §4.3 — override only with reason)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training from the latest checkpoint in --output-dir",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="run eval every N steps (higher = less frequent eval, faster run)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="save checkpoint every N steps",
    )
    args = parser.parse_args()

    # Secrets (kaggle or env)
    hf_token = get_secret("HF_TOKEN")
    wb_key = get_secret("WANDB_API_KEY")
    if wb_key:
        os.environ["WANDB_API_KEY"] = wb_key

    # Imports deferred so `--help` works without GPU deps installed
    import torch
    import wandb
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args) | {"base_model": BASE_MODEL},
    )

    print(f"Loading {BASE_MODEL} in 4-bit via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print("Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    print(f"Loading dataset {args.dataset}...")
    ds = load_dataset(args.dataset, token=hf_token)
    train_split = ds["train"]
    eval_split = ds.get("validation") or ds.get("val")
    if eval_split is None:
        raise KeyError("Dataset has no 'validation' (or 'val') split")

    # Pre-apply the Phi-3 chat template so the trainer only sees a flat "text" column.
    # This sidesteps TRL/Unsloth churn around messages-format handling and ensures
    # the *exact* tokenization used at training matches what we'll feed at inference.
    def format_row(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_split = train_split.map(
        format_row, remove_columns=train_split.column_names
    )
    eval_split = eval_split.map(
        format_row, remove_columns=eval_split.column_names
    )
    print(f"  train={len(train_split)} rows, eval={len(eval_split)} rows")
    print(f"  sample text (first 200 chars): {train_split[0]['text'][:200]!r}")

    # Pre-tokenize here (in main process, no multiprocess) so SFTTrainer's
    # internal _prepare_dataset sees an already-tokenized dataset and skips
    # its own .map() call. That map() hits a pickle error on Unsloth-patched
    # torch config objects when multiprocess gets involved.
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    print("Pre-tokenizing train split (main process, single thread)...")
    train_split = train_split.map(tokenize_fn, remove_columns=["text"])
    print("Pre-tokenizing eval split...")
    eval_split = eval_split.map(tokenize_fn, remove_columns=["text"])
    print(f"  columns after tokenization: {train_split.column_names}")

    cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to="wandb",
        packing=False,  # chat data with variable lengths — don't pack
        dataset_text_field="text",
        dataset_num_proc=1,  # avoid pickling Unsloth-patched objects across workers
        # max_seq_length is read from tokenizer.model_max_length, already set
        # to MAX_SEQ_LENGTH by FastLanguageModel.from_pretrained above.
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # TRL 0.13+ renamed `tokenizer` → `processing_class`
        args=cfg,
        train_dataset=train_split,
        eval_dataset=eval_split,
    )

    if args.resume:
        print(f"Resuming from latest checkpoint in {args.output_dir}...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Training from scratch...")
        trainer.train()

    print(f"Saving final adapter to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.hub_repo_id:
        print(f"Pushing adapter to {args.hub_repo_id}")
        trainer.model.push_to_hub(args.hub_repo_id, token=hf_token, private=True)
        tokenizer.push_to_hub(args.hub_repo_id, token=hf_token, private=True)

    wandb.finish()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
