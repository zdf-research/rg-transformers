# implementations/

One folder per member. Your folder is yours — push freely, break things freely, commit half-finished code, leave `TODO` comments everywhere. No one will judge you, and broken code pushed before the session is more useful to the group than no code at all.

## Structure

```
implementations/
└── your-name/
    ├── README.md               ← your running project notes (updated weekly)
    ├── week01_embeddings.py
    ├── week02_gradients.py
    ├── week03_attention.py
    ├── ...
    └── checkpoints/            ← saved model weights for leaderboard eval
        └── week06_final.pt
```

## Your folder README

Update `implementations/your-name/README.md` each week with what you built, any architectural choices that differ from the baseline, your current primary metric, and one thing you found surprising. By Week 13 this is your project write-up, mostly already written.

Minimum useful format:

```markdown
# your-name

| Week | Val Loss | Params | Notes                |
| ---- | -------- | ------ | -------------------- |
| 6    | 1.42     | 10.7M  | SwiGLU, RoPE preview |
```

## File naming

```
week01_embeddings.py
week02_gradients.py
week03_attention.py
week04_transformer_block.py
week05_ffn_swiglu.py
week06_full_model.py
...
```

Consistent naming makes the folder readable as a chronological project when someone (including future you) browses it in six months.

## Pushing

Commit to your personal branch `member/your-name`. No approval needed for your own folder.

```bash
git add implementations/your-name/week03_attention.py
git commit -m "week03: single-head causal attention, passes sanity check"
git push origin member/your-name
```

When you want to merge a weekly winner into `main`, open a Pull Request. See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the full workflow.
