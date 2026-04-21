# Phase 1 Walkthrough — How to Author Your Own TASK_SPEC

**Goal of this document:** Teach you to produce [TASK_SPEC.md](TASK_SPEC.md) from scratch, so you understand every line of it and can defend every decision in an interview.

**Time:** 1 hour of focused thinking. No coding. Just a text editor and this guide.

**Why this phase matters more than any other:** Every later phase — scraping, filtering, training, evaluating — derives from this spec. A sloppy spec means you'll scrape the wrong data, train on the wrong format, and discover weeks in that your eval metric doesn't measure what you care about. Each of the eight sections below is a trap beginners fall into.

---

## How to use this guide

Open [TASK_SPEC.md](TASK_SPEC.md) side-by-side. For each section below:
1. Read the **Why** — understand what problem the section solves.
2. Read the **Decisions** — the actual choices you have to make.
3. Read the **How to decide** — heuristics for picking among options.
4. Answer the **Your answer** prompt in writing. Then update `TASK_SPEC.md` to reflect *your* choice (my version is just a starting point).

If you change your mind later, you've burned days of work. So: err on the side of narrow, concrete, and testable.

---

## Section 1 — Problem statement

### Why
One sentence forces clarity. "I want to fine-tune a model on PRs" is not a problem; it's a vibe. "Given a single Python diff hunk, produce one inline review comment" is a problem — you can tell whether a given output is an attempt at it.

### Decisions you must make
- **Input unit:** a whole PR? A single file? A single hunk? One line?
- **Output unit:** a full review? One comment? A label (approve/request-changes)?
- **Turn structure:** single-shot or multi-turn?

### How to decide
- **Narrower is always better for a first project.** "Single hunk → single comment" is tractable with 4K pairs. "Whole PR → full review" needs 50K+ pairs and 32K context.
- **Pick input/output units that appear naturally in your data.** GitHub's review API gives you exactly `(hunk, inline comment)` pairs — use that granularity, don't invent new boundaries.
- **Single-shot beats multi-turn for SFT.** Multi-turn needs state and confuses the eval.

### Your answer
Write one sentence. Then write one sentence of "this is NOT":
- This model does ______.
- This model does NOT do ______.

If the "NOT" list is empty, you haven't thought about it enough. Add three items.

---

## Section 2 — Input format

### Why
The **exact string** the model sees determines what it can learn. If training strings include the filename but inference strings don't, the model fails silently. If the diff is sometimes `git diff` output and sometimes GitHub's patch format, the model learns noise instead of signal.

Formatting drift between train and inference is the single most common fine-tuning bug. Lock the format now, in characters, not in prose.

### Decisions you must make
1. **Chat template vs. raw text.** Phi-3 is instruction-tuned → use its chat template (`<|user|>...<|end|><|assistant|>...`). Skipping the template degrades quality badly.
2. **What metadata to include.** Filename? Language tag? PR title? Commit message? Each adds a cue but also a leak risk (model learns to pattern-match on title keywords instead of the diff).
3. **Diff flavor.** Unified diff with `@@` headers? Just the changed lines? Include context lines? How many?
4. **Hunk scope.** One hunk per example, or all hunks from one file concatenated?
5. **Token budget on the input side.** How many tokens can input consume before the output gets squeezed?

### How to decide
- **Strip metadata until something breaks.** Start minimal (diff only). If baseline is incoherent because the model doesn't know it's Python, add a language hint. Don't add fields "just in case."
- **Use standard unified diff with ±10 context lines.** It's what GitHub shows reviewers — the model benefits from the same context a human gets.
- **One hunk per example.** Multi-hunk rows force the model to guess which hunk the comment is about → noisy gradient.
- **Input budget ≤ 85% of context window.** Leaves room for output. Phi-3-mini-4k = 4096 tokens. Budget: input ≤ 1800, output ≤ 256, safety margin ~2000 tokens unused (training is cheaper with shorter sequences anyway).

### Your answer
Write out the **literal template string** you'll use. Copy-paste it into both your scraper and your training script later — never retype it. A single typo (`<|user|>` vs. `<|User|>`) silently breaks things.

---

## Section 3 — Output format

### Why
Three reasons you must pin this down:
1. **`max_new_tokens` needs a number.** Too low → outputs truncate mid-sentence. Too high → slow inference and rambling outputs.
2. **BERTScore penalizes length mismatch.** If references average 80 tokens and you generate 300-token essays, your metric tanks even when outputs are good.
3. **The user-facing experience is the output format.** "One focused comment" is a useful product; "a bulleted list of six vague concerns" is not.

### Decisions you must make
- Length range (min/max tokens or chars).
- Style register (terse OSS-reviewer voice? Friendly tutor? Formal?).
- Allowed structure (prose only? bullets? code blocks?).
- What the model is **forbidden** from producing.

### How to decide
- **Measure your references first.** Before picking "20–200 tokens," tokenize 100 real review comments and look at the distribution. Your output range should match the 10th–90th percentile of references.
- **Style: match the training data.** You can't make the model friendlier than its references. If OSS reviewers are terse, your model will be terse — write that into the spec rather than fighting it.
- **Forbid things aggressively.** Every "forbidden" item is a filter rule in Phase 2 (drop training rows violating it). "No LGTM-only comments" removes thousands of low-signal rows.

### Your answer
Run this quick calibration *before* writing the spec (do it in a scratch notebook once you have raw data):
```python
import statistics
lens = [len(tokenizer.encode(c)) for c in sample_comments]
print(statistics.quantiles(lens, n=10))  # deciles
```
Pick min = 10th percentile, max = 90th percentile. That's your length window.

---

## Section 4 — Scope limits

### Why
This is where good projects are made. Every scope limit is a **filter rule** that removes noise from your training set. Without scope limits, you train on a mix of bot comments, essays about code style, discussions of CI flakiness, and emoji → the model learns noise.

A tight scope with 4,000 clean pairs beats a loose scope with 40,000 noisy pairs. Always.

### Decisions you must make
For each dimension, pick **one** restrictive rule:
1. Language(s)
2. File types
3. Diff size (min and max)
4. Comment length (min and max)
5. Comment source (human only? which bots to exclude?)
6. PR status (merged only? also closed-unmerged?)
7. Comment type (inline? general PR discussion? both?)
8. Cross-references (drop anything mentioning external context?)
9. Deduplication rule
10. Per-repo cap (prevent domain collapse)

### How to decide
- **Each rule needs a one-line justification.** If you can't write why you're excluding non-Python files, you haven't thought about it. ("Python only: single-language model is tractable with 4K pairs; multi-lang would need 20K+.")
- **Write rules as predicates, not prose.** `20 <= len(comment) <= 500` is testable; "comments shouldn't be too short or too long" is not.
- **Bot blocklist: be exhaustive.** Dependabot, pre-commit.ci, CodeCov, codecov-io, github-actions, renovate-bot, mergify, sonarcloud, allcontributors. Miss one and 500 noisy rows sneak in.
- **Per-repo cap of ~15%.** Without this, one huge repo (e.g., pandas has 20K+ PRs) dominates the dataset and the model becomes a pandas-reviewer, not a Python-reviewer.

### Your answer
Write 10 rules as **exact, testable predicates** in the spec. Then, when you build the filter in Phase 2, there's a 1:1 mapping: each spec rule → one filter function → one dropped-row counter.

Example:
| Rule ID | Predicate | Why |
|---|---|---|
| F1 | `file.endswith(".py")` | Single-language scope |
| F2 | `5 <= hunk_lines <= 300` | Token budget + meaningful change |
| F3 | `20 <= len(comment) <= 500` | Drop LGTM + drop essays |

---

## Section 5 — Splits

### Why
Three splits with wrong semantics = silent leakage = fake results.
- **Train:** the model sees these and updates weights.
- **Validation:** used **during** training for LR tuning and early stopping. Touched hundreds of times.
- **Test:** touched **exactly once**, at the very end. If you peek at test during iteration, it's no longer a test.

### Decisions you must make
- Split ratios.
- Stratification key (random? by repo? by author? by date?).
- Whether to also carve out a **held-out-repo** set for generalization testing.

### How to decide
- **90/5/5 is right for small datasets.** You want most data for learning; 200 test examples is enough for BERTScore to stabilize.
- **Stratify by repo.** If `scikit-learn` is 30% of your data, it should be 30% of every split. Otherwise you'll evaluate on a skewed distribution.
- **Held-out repo set is the single highest-signal check you can do.** It catches memorization, which LoRA is prone to. Scrape 100 pairs from a repo you *exclude* from train/val/test. Evaluate both baseline and fine-tuned on it. If fine-tuned underperforms baseline here, you memorized training repos — a silent failure mode that in-distribution test won't catch.

### Your answer
Fix split sizes and stratification key in writing. Write: "Test set is opened exactly once, after training, to produce the final number. No hyperparameter or code change happens after that read." Breaking this rule later = invalidates results.

---

## Section 6 — Evaluation metrics

### Why
If you can't show a number that improved, you don't have a project. And the number has to mean something — "loss went down" doesn't count (of course it did, you trained on it).

### Decisions you must make
1. **Primary automated metric.**
2. **Manual rubric** (dimensions + scale + sample size).
3. **Qualitative artifacts** (wins, losses, failure modes).
4. **Baseline to compare against.**

### How to decide
- **BERTScore F1 over BLEU/ROUGE for text.** BLEU punishes paraphrase; BERTScore uses embedding similarity. Code review comments can say the same thing many ways — semantic similarity is the right lens.
- **Always compute a baseline on the exact same test set.** Base-model zero-shot with a strong prompt. Fine-tuned that doesn't beat zero-shot = you wasted a week. Better to find this out on day 1 than day 21.
- **Manual rubric is non-optional.** BERTScore can be gamed by generic filler. Reading 50 outputs yourself is the only way to tell.
- **Pick rubric dimensions from the task, not from a template.** For review comments: relevance, actionability, factuality. For a summarization task these'd be different.
- **Rubric levels 1/3/5, not 1-10.** Humans can't distinguish 10 levels consistently. Three anchor points with explicit descriptions → consistent scoring.

### Your answer
Write:
- Primary metric + model used to compute it (e.g., "BERTScore F1, scorer = `microsoft/deberta-xlarge-mnli`").
- Rubric table: 3 dimensions × 3 anchor levels, each anchor with a concrete description.
- Report format: baseline vs. fine-tuned side-by-side, plus delta.

---

## Section 7 — Success criteria

### Why
Decide in advance what "good enough to ship" means. Without this, you'll either ship a mediocre result (confirmation bias) or tune forever (perfectionism). Write the bar down before you know the result.

### Decisions you must make
- Minimum acceptable BERTScore improvement over baseline.
- Minimum held-out-repo score.
- Minimum manual rubric mean.
- Minimum format-compliance rate.
- What to do if any criterion fails.

### How to decide
- **10% relative improvement is a realistic floor for a ~4K-pair SFT run.** Below that, the noise/signal ratio suggests the fine-tune didn't learn much.
- **Held-out-repo within 5% of in-domain.** Wider gap = overfit to training repos.
- **Relevance ≥ 3.5/5 is a sensible bar.** 3/5 = "touches the right area but misses the specific issue" — below that isn't useful.
- **Format compliance ≥95%.** If the model violates your output format 10% of the time (too long, wrong structure), post-processing hacks start to pile up — smell of a bad training run.

### Your answer
Write the exact thresholds. Then write what you'll do if each fails (the decision, not a vague "investigate"):
- "If BERTScore delta <10% → stop, re-examine dataset quality before retraining."
- "If held-out-repo gap >5% → reduce epochs to 1, add dropout 0.05, retrain."
- "If format compliance <95% → audit training-data formatting for drift, retrain."

---

## Section 8 — Non-goals

### Why
Non-goals prevent scope creep — both your own ("wouldn't it be cool if…") and from reviewers/stakeholders. Writing them down lets you say "that's explicitly out of scope, per the spec" without re-litigating.

### Decisions you must make
List the 5–10 things people might reasonably assume you're doing, and state that you are not.

### How to decide
Brainstorm. "What would a hiring manager plausibly ask 'does your model do X?'" Each of those gets a line.

### Your answer
Aim for 5+ non-goals. My starter list:
- Not catching real bugs at scale.
- Not beating frontier models.
- Not multi-language.
- Not whole-PR reasoning.
- Not RLHF/DPO.

Add your own. If the list is short, you're not thinking about edge cases.

---

## How to actually sit down and do this (practical sequence)

1. **(10 min) Open [TASK_SPEC.md](TASK_SPEC.md).** Read it once end-to-end. Don't edit yet.
2. **(10 min) Section 1.** Write your one-sentence problem statement and three "NOT" items. Paste them in.
3. **(10 min) Sections 2 + 3.** Write the literal chat template string. Skip the reference-length calibration for now (noted as a TODO for Phase 2, once raw data exists).
4. **(15 min) Section 4.** List your 10 scope rules as predicates. This is the longest part; most of the thinking lives here.
5. **(5 min) Section 5.** Pick split ratios, stratification key, and commit to the "test touched once" discipline in writing.
6. **(5 min) Section 6.** Pick BERTScore variant, rubric dimensions, rubric anchors.
7. **(5 min) Section 7.** Write the four thresholds and the four "if X fails then Y" actions.
8. **(5 min) Section 8.** Write 5+ non-goals.
9. **(5 min) Re-read end-to-end.** Does every section constrain a later phase in a way you can implement? If a section is prose without testable rules, rewrite it.

Total: ~70 min. Worth every minute.

---

## Red flags that your spec isn't ready

Scan your spec for these before closing the file:

- [ ] Any sentence containing "should" without a measurable criterion. ("Output should be high quality" → delete or quantify.)
- [ ] Any rule that can't be expressed as code (`def filter(row) -> bool`).
- [ ] An empty non-goals section.
- [ ] No held-out-repo plan.
- [ ] No baseline comparison in the eval section.
- [ ] A length constraint that wasn't calibrated against real data (flag it as a TODO, re-visit after scraping).
- [ ] A chat template you haven't copy-pasted into a scratch file to visually confirm special tokens are right.

If any are present, fix them before moving to Phase 2. Cheap now, expensive later.

---

## What "done" looks like

You close Phase 1 when:
1. [TASK_SPEC.md](TASK_SPEC.md) is filled in with your choices (not just the starter content).
2. Every section has at least one testable, measurable statement.
3. You could hand it to another engineer and they'd build the same dataset you would.
4. You committed it to git (so Phase 2 changes can be diffed against it).

Then, and only then, move to Phase 2 (`scripts/scrape_prs.py`).
