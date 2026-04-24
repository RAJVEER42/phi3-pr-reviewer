#!/usr/bin/env python3
"""Recovery script — compute BERTScore from a saved predictions JSONL.

Use when run_eval.py generated predictions but BERTScore failed (e.g., the
bert-score package wasn't installed in the notebook). Avoids re-generating.

The predictions JSONL should have one row per example with at minimum:
    {"idx": ..., "prediction": "...", "reference": "...", ...}

Usage:
    pip install bert-score
    python scripts/score_predictions.py \\
        --predictions /kaggle/working/finetuned_predictions.jsonl \\
        --tag finetuned \\
        --out-dir /kaggle/working
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCORER_MODEL = "microsoft/deberta-xlarge-mnli"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="path to predictions JSONL")
    parser.add_argument("--tag", required=True, help="run name for output filenames")
    parser.add_argument("--out-dir", default="/kaggle/working")
    parser.add_argument("--scorer-model", default=SCORER_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"Predictions file not found: {pred_path}", file=sys.stderr)
        return 1

    rows = []
    with pred_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        print("Predictions file is empty.", file=sys.stderr)
        return 1

    predictions = [r["prediction"] for r in rows]
    references = [r["reference"] for r in rows]
    print(f"Loaded {len(rows)} (prediction, reference) pairs")

    print(f"Computing BERTScore with {args.scorer_model}...")

    # Workaround for bert-score + newer transformers: some tokenizers (e.g.,
    # microsoft/deberta-xlarge-mnli) ship with model_max_length=VERY_LARGE_INT
    # which overflows Rust's i64 when passed to enable_truncation. Monkey-patch
    # AutoTokenizer.from_pretrained to cap model_max_length at 512 for the
    # duration of the bert_score call.
    from transformers import AutoTokenizer
    from bert_score import score as bert_score

    _orig_from_pretrained = AutoTokenizer.from_pretrained

    def _patched_from_pretrained(*a, **kw):
        tok = _orig_from_pretrained(*a, **kw)
        if getattr(tok, "model_max_length", 0) > 10_000_000:
            tok.model_max_length = 512
        return tok

    AutoTokenizer.from_pretrained = _patched_from_pretrained
    try:
        P, R, F = bert_score(
            predictions,
            references,
            model_type=args.scorer_model,
            lang="en",
            batch_size=args.batch_size,
            verbose=True,
        )
    finally:
        AutoTokenizer.from_pretrained = _orig_from_pretrained
    mean_f = F.mean().item()
    mean_p = P.mean().item()
    mean_r = R.mean().item()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_preds = out_dir / f"{args.tag}_predictions.jsonl"
    out_summary = out_dir / f"{args.tag}_summary.json"

    with out_preds.open("w") as f:
        for i, row in enumerate(rows):
            f.write(
                json.dumps(
                    {
                        **row,
                        "bertscore_f1": F[i].item(),
                        "bertscore_precision": P[i].item(),
                        "bertscore_recall": R[i].item(),
                    }
                )
                + "\n"
            )

    with out_summary.open("w") as f:
        json.dump(
            {
                "tag": args.tag,
                "scorer": args.scorer_model,
                "n_examples": len(rows),
                "bertscore_f1": mean_f,
                "bertscore_precision": mean_p,
                "bertscore_recall": mean_r,
            },
            f,
            indent=2,
        )

    print(f"\n  BERTScore F1: {mean_f:.4f}")
    print(f"  Predictions: {out_preds}")
    print(f"  Summary:     {out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
