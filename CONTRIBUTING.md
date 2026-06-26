# Contributing to the Reading Group Repo

> _This document exists so that Week 1 is not derailed by a merge conflict on the shared tokeniser. Please read it once. It is not long._

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Getting Started](#getting-started)
3. [Weekly Workflow](#weekly-workflow)
4. [Pushing Your Implementation](#pushing-your-implementation)
5. [Pushing Your Presentation](#pushing-your-presentation)
6. [The Common Module](#the-common-module)
7. [The Leaderboard](#the-leaderboard)
8. [Branch & Commit Conventions](#branch--commit-conventions)
9. [Pull Request Rules](#pull-request-rules)
10. [Environment Setup](#environment-setup)
11. [Using the H100 Cluster](#using-the-h100-cluster)

---

## Repository Structure

```
rg-transformers/
├── README.md                        ← the curriculum and group manifesto
├── CONTRIBUTING.md                  ← this document
├── LEADERBOARD.md                   ← updated after each session
│
├── common/                          ← shared utilities (protected, PR required)
│   ├── tokenizer.py                 ← character-level and BPE tokenisers
│   ├── data.py                      ← dataset loading (TinyShakespeare, WikiText)
│   ├── eval.py                      ← evaluation script — do not modify without a PR
│   ├── train.py                     ← reference training loop
│   └── pyproject.toml               ← project metadata and pinned dependencies (managed with uv)
│
├── presentations/
│   ├── week01-alice/
│   │   ├── slides.pdf               ← exported slides (PDF preferred)
│   │   └── notes.md                 ← optional speaker notes or extra references
│   └── week03-bob/
│       └── slides.pdf
│
└── implementations/
    ├── alice/
    │   ├── README.md                ← alice's architecture notes and results
    │   ├── week01_embeddings.py
    │   ├── week03_attention.py
    │   └── ...
    └── bob/
        └── ...
```

**The key rule:** `common/` is shared and protected. `implementations/your-name/` is yours — push freely, break things freely, no one will judge you.

---

## Getting Started

### 1. Admin onboarding

An admin will run the onboarding GitHub Action for you. This creates your `member/<username>` branch and scaffolds your implementation folder automatically. You do not need to create anything yourself.

### 2. Clone and switch to your branch

```bash
git clone https://github.com/zdf-research/rg-transformers.git
cd rg-transformers
git pull
git switch member/your-name
```

Your branch already contains `implementations/your-name/README.md`. Add a one-line description of yourself there. This becomes your project page by Week 13.

> **Branch protection:** Pushing directly to `main` is disabled for everyone. All code lands via Pull Request.

### 3. Set up your environment

We use `uv` for dependency management. Once the shared `pyproject.toml` is in place:

```bash
uv sync
# or, for editable install of the local package:
uv pip install -e .
```

Until then, follow any setup instructions posted in the group chat.

See [Environment Setup](#environment-setup) for conda/venv instructions and GPU configuration.

### 4. Verify common utilities work

```bash
python common/eval.py --help
python -c "from common.data import load_tinshakespeare; print('ok')"
```

If either of these fails, open an issue before touching anything else.

---

## Weekly Workflow

Here is the expected rhythm each week:

```
Monday / Tuesday      Read the core references for the week
Wednesday / Thursday  Write your implementation module
Friday (before EOD)   Push your code to the repo   ← hard deadline
Weekend               Optional: run eval, update leaderboard entry
Session day           Come prepared to discuss and show your code
```

The **Friday push deadline** is the only hard rule in this document. It exists so that the person presenting has time to look at everyone's code before the session, and so the code lab is a discussion rather than a live debugging session (though live debugging sessions are also fine and often more instructive).

---

## Pushing Your Implementation

### File naming

Name your files by week number and topic:

```
week01_embeddings.py
week02_rnn_lstm.py
week03_attention.py
week04_transformer_block.py
...
```

This makes the folder readable as a chronological project by Week 13.

### Push to your member branch

All your work lives on your `member/your-name` branch. Push freely — no approval needed for your own folder:

```bash
git add implementations/your-name/week03_attention.py
git commit -m "week03: single-head causal attention, passes sanity check"
git push origin member/your-name
```

When you are ready to land code in `main` (e.g. a weekly winner merge, or a presentation), open a Pull Request.

### Keep a running README in your folder

Update `implementations/your-name/README.md` each week with:

- What you implemented
- Any architectural choices you made that differ from the baseline
- Your current primary metric for the week
- One thing you found surprising or confusing

By Week 13 this is your project write-up, mostly already written.

### It is fine to push broken code

Seriously. A file that raises a `NotImplementedError` or has a known bug is more useful to the group than no file. Add a comment at the top:

```python
# STATUS: forward pass works, backward pass untested — Week 3
# Known issue: causal mask is wrong for batch_size > 1
```

---

## Pushing Your Presentation

Export your slides to PDF before pushing. This ensures they render correctly for everyone regardless of what you used to make them (PowerPoint, Keynote, Google Slides, Beamer, napkin).

```bash
mkdir -p presentations/week03-your-name
cp your_slides.pdf presentations/week03-your-name/slides.pdf
git add presentations/week03-your-name/
git commit -m "week03: add attention presentation slides"
git push origin member/your-name
```

Optionally add a `notes.md` with links, extended derivations, or things you wanted to cover but ran out of time.

---

## The Common Module

`common/` contains the shared infrastructure everyone's implementation depends on. Changes here can break everyone's code simultaneously, which is annoying. So:

- **Read before you use.** Skim `common/eval.py` and `common/data.py` before Week 1 so you know what they do.
- **Do not modify `common/` directly.** All changes go through a pull request with at least one approval from another group member.
- **If something is broken or missing, open an issue first.** Describe what you need. Someone else may already have a fix.
- **The eval script is sacred.** `common/eval.py` defines the leaderboard metric. Changes to it invalidate all historical leaderboard entries. Treat it like production code.

### Proposing a change to common/

```bash
git checkout -b fix/tokenizer-bpe-edge-case
# make your changes
git commit -m "fix: handle empty sequence in BPE tokeniser"
git push origin fix/tokenizer-bpe-edge-case
# open a PR on GitHub, tag at least one person for review
```

PR description should include: what was broken, what you changed, and a test showing it works.

---

## The Leaderboard

The leaderboard lives in `LEADERBOARD.md` and is updated after each session. Each week has its own metric(s) defined in [CURRICULUM.md](CURRICULUM.md). The winner is determined by the best score on the week's primary metric.

### Running the eval

```bash
# Example for Week 6 (validation loss is the metric)
python common/eval.py \
  --checkpoint implementations/your-name/checkpoints/week06_final.pt \
  --name "your-name" \
  --week 6
```

This prints your metrics. Hand the output to whoever is maintaining the board that week, or open a PR against `LEADERBOARD.md`.

### Leaderboard format

```markdown
| Week | Name  | Primary Metric | Secondary Metric | Notes                        |
| ---- | ----- | -------------- | ---------------- | ---------------------------- |
| 6    | alice | 1.423          | 10.7M params     | SwiGLU, RoPE preview         |
| 6    | bob   | 1.551          | 10.4M params     | Vanilla FFN, learned pos emb |
```

### Rules

- Only submit results from the **shared eval script** on the **fixed dataset split**. Do not submit results from your own eval loop — the point is comparability.
- Secondary metrics are optional but encouraged.
- If your result seems suspiciously good, the group will ask you to walk through your code. This is not an accusation — it is how we catch bugs (including the bug where you accidentally eval on the training set, which happens to everyone once).

### Weekly winner selection

At the end of each session, the group reviews the leaderboard for that week. The implementation with the **best primary metric** is selected as the weekly winner. The winner opens a PR to merge their module into `group_project/`. See [CURRICULUM.md](CURRICULUM.md) for per-week metrics.

In case of ambiguity (e.g. a faster but slightly less accurate model), the group votes. Clarity and correctness are tie-breakers.

---

## Branch & Commit Conventions

### Branch naming

| Purpose                    | Format                     | Example                     |
| -------------------------- | -------------------------- | --------------------------- |
| Personal work branch       | `member/your-name`         | `member/alice`              |
| Fix to common utilities    | `fix/short-description`    | `fix/eval-bfloat16-cast`    |
| Feature addition to common | `feat/short-description`   | `feat/wikitext-data-loader` |
| Weekly winner merge        | `merge/weekNN-winner`      | `merge/week05-swiglou`      |

Everyone works on a persistent `member/your-name` branch. All changes to `main` go through a Pull Request — even for your own implementation folder.

### Commit messages

Keep them short and useful. The format is:

```
weekNN: one-line description of what this commit does
```

Examples of good commit messages:

```
week03: add causal masking, fix off-by-one in mask shape
week05: swiglu working, hidden dim scaled to 2/3
week08: gqa with kv cache, ~3x faster generation
```

Examples of unhelpful commit messages:

```
update
fix
WIP
asdfgh
```

We are all guilty of the last one on a Friday afternoon. Try anyway.

---

## Pull Request Rules

All changes to `main` go through a Pull Request. This includes weekly winner merges, presentation uploads, and fixes to `common/`.

### PR checklist

- [ ] Describe what the change does and why in the PR description
- [ ] Include a minimal test or usage example showing it works
- [ ] Tag at least one other group member as reviewer
- [ ] Do not merge your own PR — wait for one approval
- [ ] If the change touches `eval.py`, tag everyone and wait for group consensus

### Review expectations

When you are tagged as a reviewer, aim to respond within 48 hours. A review does not need to be exhaustive — a "looks good, tested locally" with a note on anything you noticed is sufficient. The goal is a second pair of eyes, not a formal audit.

---

## Environment Setup

We use [`uv`](https://docs.astral.sh/uv/) for dependency management.

### Quick start (once pyproject.toml is available)

```bash
uv sync
# or, for an editable install:
uv pip install -e .
```

### Verifying GPU access

```python
import torch
print(torch.cuda.is_available())          # should be True
print(torch.cuda.get_device_name(0))      # should name your GPU
print(torch.cuda.memory_allocated() / 1e9, "GB")
```

### For H100 cluster jobs

See [Using the H100 Cluster](#using-the-h100-cluster) below, or [COMPUTE.md](COMPUTE.md) for the full guide. Do not run cluster jobs for experiments that fit on a local GPU — save the H100s for things that actually need them.

---

## Using the H100 Cluster

The cluster is a shared resource. Be a good citizen.

### Before launching any job

1. **Test locally first.** Run one step with `--max_steps 5` on your machine. If it crashes, fix it before submitting.
2. **Set a compute budget.** Every job should have a `--max_steps` or `--max_hours` limit. No open-ended runs.
3. **Check cluster utilisation.** If the cluster is busy with someone else's long run, wait or use a cloud alternative.

### Job submission template

A minimal SLURM script for the cluster:

```bash
#!/bin/bash
#SBATCH --job-name=readinggroup-week09
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00          # set a real limit — do not use 24:00:00 as default
#SBATCH --output=logs/%j.out

source ~/.bashrc
# if using uv on the cluster:
uv run python implementations/your-name/week09_flashattn_benchmark.py \
  --batch_size 8 \
  --seq_len 4096 \
  --max_steps 500
```

Save your job scripts in `implementations/your-name/cluster/` so others can learn from them.

### Logging runs

Use Weights & Biases (free tier) or TensorBoard. A run without logs is a run that did not happen.

```python
import wandb
wandb.init(project="zdf-research", name="alice-week09-flashattn")
wandb.log({"train/loss": loss, "train/step": step})
```

Set your W&B entity to the group organisation so runs are visible to everyone.

### If you accidentally launch something expensive

Kill it. `scancel <job_id>`. No shame. The shame is in letting it run for six hours out of sunk-cost reasoning.

---

## A Final Note

This document will inevitably be wrong about something by Week 3. When that happens, open a PR and fix it. The repo, like the group, is a living thing.

---

_Questions? Open an issue. Broken something? Open an issue. Found something interesting? Open an issue. Issues are free._
