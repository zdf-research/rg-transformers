# common/

Shared infrastructure that every member's implementation depends on. Treat this folder like production code — changes here can break everyone's setup simultaneously.

## Contents

| File               | Purpose                                                                                                                                                              |
| :----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tokenizer.py`     | Character-level tokeniser (for early weeks) and shared GPT-2 style BPE tokeniser (used for all leaderboard evaluation)                                               |
| `data.py`          | Dataset loading for TinyShakespeare and the fixed WikiText-103 held-out split                                                                                        |
| `eval.py`          | **The leaderboard script.** Computes validation cross-entropy on the fixed eval split. Do not modify without group consensus — changes invalidate historical entries |
| `train.py`         | Reference training loop. Not authoritative — use it as a starting point or ignore it entirely                                                                        |
| `pyproject.toml`   | Project metadata and dependencies (managed with `uv`). If you need to add a package, open an issue first                                                            |

### Data Management (`data.py`)
To keep the Git repository lightweight and clone speeds fast, we **do not** commit datasets to version control. Instead, `data.py` fetches the required corpora (e.g., TinyShakespeare, WikiText) on demand via HTTPS and caches them locally in a `data/` folder at the project root (ignored by `.gitignore`). This ensures reproducibility (everyone pulls from the exact same URL) while avoiding Git history bloat.

## Rules

**Do not push directly to `common/`.** All changes go through a pull request with at least one approval. If something is broken or missing, open an issue before writing a fix — someone else may already be working on it.

The eval script in particular is effectively immutable once the leaderboard starts. If you find a genuine bug in it, open an issue, tag everyone, and wait for group discussion before touching it.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full PR process.
