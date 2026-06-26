# rg-transformers

> _Because at some point, someone has to build the thing that makes the dashboards obsolete._

A 13-week reading and build group. We construct a nanoGPT-style transformer from first principles — every line written, every design decision traced to a source.

---

## What you will build

- **Weeks 1–6:** A working character-level transformer, trained and generating text
- **Weeks 7–12:** The full modern stack — RoPE, FlashAttention, mixed precision, distributed training, efficient generation
- **Week 12:** Merge weekly winners into the group project
- **Week 13:** Final demo day, group model presentation, and aperitivo

Every week, one implementation is voted "winner" and merged into the shared group project. By the end, the group owns one fully-featured transformer, built from its own best ideas.

---

## Quickstart

1. **Get onboarded by an admin**  
   An admin will run the GitHub Action to create your `member/<username>` branch and scaffold your folder. Just ask.

2. **Clone and switch to your branch**
   ```bash
   git clone https://github.com/zdf-research/rg-transformers.git
   cd rg-transformers
   git pull
   git switch member/your-name
   ```

3. **Set up the environment** (using `uv`)
   ```bash
   uv sync             # once pyproject.toml is available
   # or for now:
   uv pip install -e .
   ```

4. **Pre-work before Week 1**  
   Watch: [Andrej Karpathy — "Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) (~2 hours). Mandatory. You should be able to sketch the forward pass on a napkin by the first session.

---

## Quick Links

| Resource | Description |
| :------- | :---------- |
| [CURRICULUM.md](CURRICULUM.md) | Full 13-week schedule, per-week metrics, and winner criteria |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Git workflow, member branches, and `common/` rules |
| [COMPUTE.md](COMPUTE.md) | H100 cluster guide, SLURM templates, and responsible use |
| [LEADERBOARD.md](LEADERBOARD.md) | Weekly metric results and running scores |
| [FUTURE_TRACKS.md](FUTURE_TRACKS.md) | Post-cohort research directions (MoE, RLHF, ViT, etc.) |

---

## How Sessions Work

- **When:** Weekly, ~90 minutes
- **Format:**
  1. **Paper & Concept Presentation** (45 min) — one member teaches the core idea back to the group
  2. **Code Review Lab** (45 min) — walk through implementations, debug together, log metrics
- **Deadline:** Push your code to `member/your-name` by Friday EOD. Broken code welcome. (See [CONTRIBUTING.md](CONTRIBUTING.md))

---

## Norms

- **Come prepared.** Read the core references before the session.
- **Push broken code.** A file with a `NotImplementedError` and a comment is more useful than no file.
- **Ask why, not just how.** If you can run it but cannot explain the gradient flow, that is the discussion.
- **Be kind about code.** Review is for understanding, not judgement.
- **Missing a session is fine; missing two in a row means a check-in.** Momentum is the only real risk.

---

_Last updated: Week 0. See you on the other side of the pivot table._
