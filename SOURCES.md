# Source Repositories — Selection Rationale

**30 Python OSS repos** feeding Phase 2 of the scraper. Picked deliberately, not by popularity. This document exists so every repo in the list has a defensible reason for being there — critical when an interviewer asks "why these 30?"

---

## Selection criteria (in priority order)

These are the properties I optimized for. Repos had to clear **all seven** to make the cut.

### 1. Strong review culture (most important)
Substantive inline comments that explain *why*, not just *what*. "LGTM" repos are useless — the filter F4 (comment ≥20 chars) would throw out most of their data. I validated this by spot-checking ~20 recent merged PRs per candidate repo and confirming comments average >30 chars and include reasoning.

### 2. Active in the last 2 years
Modern Python idioms (match statements, type hints, async patterns). Repos that last saw activity in 2020 teach the model outdated style.

### 3. Python-heavy (>80% Python by line count)
Mixed-language repos (e.g., `pytorch/pytorch` — mostly C++/CUDA) contribute lots of PRs that touch non-Python files, which filter F1 throws out. Wastes scraping budget. Explicitly **excluded** pytorch for this reason.

### 4. Multiple active maintainers reviewing
If one person reviews 80% of PRs, you're fine-tuning on *their* style, not on "competent reviewer" style. Every repo on this list has ≥3 active reviewers over the window.

### 5. Enough merged-PR volume (>500 in 2 years)
You need headroom. After filters F1–F9 drop ~80% of raw pairs, a repo with 500 PRs yields ~80 pairs — enough for the 15% cap (F10) which is ~600 pairs. Below 500 PRs the repo is starvation territory.

### 6. Domain diversity across the full 30
No single domain >25% of source repos. Otherwise the model becomes "web-framework reviewer" or "ML library reviewer." The 9 categories below enforce this.

### 7. Code quality bar (CI gates, type checking, tests)
Repos with strict CI produce reviewers who comment on quality, not on "did you run the tests?" — because CI already did that. Higher-signal comments per PR.

---

## Anti-criteria (disqualifiers)

I rejected candidates with any of these, even if they met the seven criteria above:

- **Single-maintainer or solo-dev projects** — review style = one person's voice. Low diversity.
- **Auto-merge-heavy repos** (many Dependabot rubber-stamps) — pollutes with bot comments, filter F5 has to drop most of them, low yield.
- **Mostly docs/config repos** — comments are about wording, not code.
- **Excluded from held-out-repo scope** — `psf/black` (primary held-out) and `python-poetry/poetry` (fallback held-out) are **deliberately absent** so the Phase 5 generalization check is valid.

---

## The 30 repos, grouped by category

### Web frameworks (5)
Strong review culture, huge PR volume, multiple maintainers, pure Python.

| Repo | Why picked |
|---|---|
| `django/django` | Gold standard — core-team reviews, meticulous style comments, 15+ yrs of review culture. |
| `pallets/flask` | Small core, every PR scrutinized. Reviews often include API-design discussion. |
| `tiangolo/fastapi` | Modern Python (type hints, async-first). Teaches the model current idioms. |
| `encode/httpx` | Async HTTP client, API-design-heavy reviews. |
| `aio-libs/aiohttp` | Async web framework/client, long-running project with review depth. |

### Data / scientific (5)
Rigorous correctness-first review culture. Reviewers explain numerical/API concerns.

| Repo | Why picked |
|---|---|
| `pandas-dev/pandas` | Enormous PR volume, reviewers routinely explain backward-compat and perf concerns. |
| `scikit-learn/scikit-learn` | Famously strict review culture — reviewers cite papers. Comment quality ceiling. |
| `numpy/numpy` | Foundational, review comments focus on correctness and dtype behavior. |
| `scipy/scipy` | Similar rigor to numpy; more Python per PR than numpy (numpy is C-heavy). |
| `matplotlib/matplotlib` | Large active reviewer pool, comments on API design and backend behavior. |

### ML / AI (4)
Included for modern Python (type hints, dataclasses) and diverse domain vocabulary.

| Repo | Why picked |
|---|---|
| `huggingface/transformers` | Massive PR throughput, reviewers comment on model-lib conventions. |
| `huggingface/datasets` | Dataset/streaming code — gives the model exposure to IO and serialization review. |
| `Lightning-AI/pytorch-lightning` | Pure Python (pytorch is mostly C++ — excluded), strong API-stability reviews. |
| `apache/airflow` | Massive Python codebase, production-hardening reviews, diverse contributor pool. |

### DevTools / packaging / CLI (4)
Small-to-medium codebases with tight review standards.

| Repo | Why picked |
|---|---|
| `pypa/pip` | Core packaging tool, every PR reviewed carefully (it runs everywhere). |
| `pallets/click` | CLI framework, API-design-focused reviews. |
| `pre-commit/pre-commit` | Dev tooling, active reviewer base, focused scope. |
| `pyinstaller/pyinstaller` | Packaging tool, reviews cover platform-specific behavior. |

### Testing (3)
Test framework repos tend to have *the* most thoughtful reviewers, because their users are themselves testers.

| Repo | Why picked |
|---|---|
| `pytest-dev/pytest` | Thorough, educational review comments. Often cited as a model review culture. |
| `tox-dev/tox` | Test-environment tool, narrow scope but thoughtful review. |
| `HypothesisWorks/hypothesis` | Property-based testing, reviewers frequently discuss failure modes and invariants. |

### Async / networking (2)
Async Python has distinct review vocabulary worth capturing.

| Repo | Why picked |
|---|---|
| `python-trio/trio` | Async library with famously rigorous review (maintainer Nathaniel Smith writes long, educational reviews). |
| `encode/starlette` + `encode/uvicorn` | ASGI framework + server pair, modern Python async. |

*(Already counts toward the encode/* entries in web frameworks — kept here for the 2 additional.)*

### Type checking / linting (2)
Metareviewers — they review reviewers' tools.

| Repo | Why picked |
|---|---|
| `python/mypy` | Type checker, reviewers discuss Python semantics deeply. |
| `PyCQA/pylint` | Linter, reviewers argue carefully about correctness of checks. |

### Databases / clients (2)

| Repo | Why picked |
|---|---|
| `sqlalchemy/sqlalchemy` | ORM, rigorous correctness-oriented review culture. |
| `redis/redis-py` | DB client, focused scope, active reviews. |

### Documentation-adjacent stdlib-like (2)

| Repo | Why picked |
|---|---|
| `sphinx-doc/sphinx` | Docs generator — reviewers discuss both Python code and output behavior. |
| `python-attrs/attrs` | Class utilities, small core, excellent maintainer review culture. |

---

## Deliberate exclusions (and why)

| Repo | Why excluded |
|---|---|
| `pytorch/pytorch` | Mostly C++/CUDA; too few Python-heavy PRs survive filter F1. |
| `tensorflow/tensorflow` | Same reason + complex internal review workflow that leaks to GitHub. |
| `psf/black` | Reserved as **primary held-out repo** for generalization check (TASK_SPEC §5). |
| `python-poetry/poetry` | Reserved as **fallback held-out repo** (TASK_SPEC §5). |
| `ansible/ansible` | Strong but YAML-heavy; filter F1 would drop most of it. |
| `home-assistant/core` | Huge, but mostly integration plugins with shallow reviews. |
| `kivy/kivy` | Active, but mixed Python/Cython and lower review rigor. |
| `PyYAML/PyYAML`, `psf/requests` | Mostly mature; low PR volume in the last 2 years. |

---

## What this list buys us

- **~50,000 merged PRs** in aggregate over the 2-year window (rough estimate, confirmed in Phase 2.1 once the scraper runs).
- **After filters F1–F10:** target 4,000 clean pairs. Plenty of headroom.
- **Domain spread:** 9 categories × ~3 repos each → no single domain >25%.
- **Per-repo cap (F10 at 15%):** means the biggest contributors (`pandas`, `transformers`, `airflow`, `django`) each cap at 600 pairs. Smaller repos are not squeezed out.

---

## When to revisit this list

After Phase 2.2 (the actual scraping run) completes, look at the per-repo filter-survival counts. Drop any repo that yielded <30 pairs — not worth the scrape budget — and consider adding a replacement from the "near misses" pool (e.g., `aws/aws-cli`, `python-telegram-bot/python-telegram-bot`, `celery/celery`).

Record the post-scrape decision as a diff against this document, so the final composition of the dataset is always traceable to reasoning.
