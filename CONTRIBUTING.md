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
│   └── requirements.txt             ← pinned dependencies for reproducibility
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

### 1. Fork or clone the repo

You do not need to fork — everyone has write access to the organisation repo. Just clone directly:

```bash
git clone https://github.com/zdf-research/rg-transformers.git
cd rg-transformers
```

### 2. Create your implementation folder

Do this once, in your first week:

```bash
mkdir -p implementations/your-name
touch implementations/your-name/README.md
```

Add a one-line description of yourself to your `README.md`. This becomes your project page by Week 12.

### 3. Set up your environment

```bash
pip install -r common/requirements.txt
```

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

This makes the folder readable as a chronological project by Week 12.

### Commit directly to main (for your own folder only)

Since your `implementations/your-name/` folder is yours, you can commit directly to `main` for your own work:

```bash
git add implementations/your-name/week03_attention.py
git commit -m "week03: single-head causal attention, passes sanity check"
git push origin main
```

No PR needed. No approval needed. Just push.

### Keep a running README in your folder

Update `implementations/your-name/README.md` each week with:

- What you implemented
- Any architectural choices you made that differ from the baseline
- Your current validation loss (copy from `eval.py` output)
- One thing you found surprising or confusing

By Week 12 this is your project write-up, mostly already written.

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
git push origin main
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

The leaderboard lives in `LEADERBOARD.md` and is updated after each session. The metric is **validation cross-entropy loss** on a fixed held-out split of WikiText-103, computed by `common/eval.py`.

### Running the eval

```bash
python common/eval.py \
  --checkpoint implementations/your-name/checkpoints/week06_final.pt \
  --name "your-name" \
  --week 6
```

This prints your loss and appends a row to a local `leaderboard_update.json`. Paste the output into the leaderboard update PR after each session, or hand it to whoever is maintaining the board that week.

### Leaderboard format

```markdown
| Week | Name  | Val Loss | Params | Notes                        |
| ---- | ----- | -------- | ------ | ---------------------------- |
| 6    | alice | 1.423    | 10.7M  | SwiGLU, RoPE preview         |
| 6    | bob   | 1.551    | 10.4M  | Vanilla FFN, learned pos emb |
```

### Rules

- Only submit results from the **shared eval script** on the **fixed dataset split**. Do not submit results from your own eval loop — the point is comparability.
- Secondary metrics (params, tokens/sec, training time) are optional but encouraged from Week 7 onwards.
- If your loss seems suspiciously good, the group will ask you to walk through your code. This is not an accusation — it is how we catch bugs (including the bug where you accidentally eval on the training set, which happens to everyone once).

---

## Branch & Commit Conventions

### Branch naming

| Purpose                    | Format                     | Example                     |
| -------------------------- | -------------------------- | --------------------------- |
| Your weekly implementation | `impl/your-name/weekNN`    | `impl/alice/week03`         |
| Fix to common utilities    | `fix/short-description`    | `fix/eval-bfloat16-cast`    |
| Feature addition to common | `feat/short-description`   | `feat/wikitext-data-loader` |
| Presentation               | just push to main directly | —                           |

You only need branches if you are touching `common/`. For your own implementation folder, push to `main` directly.

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

PRs are only required for changes to `common/`. For everything else, push directly.

### PR checklist (for common/ changes)

- [ ] Describe what the change does and why in the PR description
- [ ] Include a minimal test or usage example showing it works
- [ ] Tag at least one other group member as reviewer
- [ ] Do not merge your own PR — wait for one approval
- [ ] If the change touches `eval.py`, tag everyone and wait for group consensus

### Review expectations

When you are tagged as a reviewer, aim to respond within 48 hours. A review does not need to be exhaustive — a "looks good, tested locally" with a note on anything you noticed is sufficient. The goal is a second pair of eyes, not a formal audit.

---

## Environment Setup

### Option 1: pip + venv (simplest)

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r common/requirements.txt
```

### Option 2: conda

```bash
conda create -n readinggroup python=3.11
conda activate readinggroup
pip install -r common/requirements.txt
```

### Verifying GPU access

```python
import torch
print(torch.cuda.is_available())          # should be True
print(torch.cuda.get_device_name(0))      # should name your GPU
print(torch.cuda.memory_allocated() / 1e9, "GB")
```

### For H100 cluster jobs

See [Using the H100 Cluster](#using-the-h100-cluster) below. Do not run cluster jobs for experiments that fit on a local GPU — save the H100s for things that actually need them.

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
conda activate readinggroup

python implementations/your-name/week09_flashattn_benchmark.py \
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
