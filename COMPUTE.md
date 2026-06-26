# Compute Guide

We have access to a cluster of H100 GPUs. The rule is simple: **use it thoughtfully, not wastefully**. The 13-week curriculum is designed so that most experimentation runs comfortably on a single consumer GPU or even CPU for the early weeks. The H100s are for moments where scale actually teaches you something new.

---

## When to Use the Cluster

| Phase | Recommended Compute | Why |
| :---- | :------------------ | :-- |
| Weeks 1–5 | Local CPU / single GPU | Models are tiny; fast iteration matters more than scale |
| Weeks 6–8 | Single H100 or cheap cloud GPU | First full training runs; KV cache profiling benefits from real VRAM |
| Weeks 9–12 | H100 (1–2 GPUs) | FlashAttention benchmarks, mixed precision, distributed training |
| Post-curriculum | H100 cluster (multi-GPU) | DDP, tensor parallelism, larger models — see [FUTURE_TRACKS.md](FUTURE_TRACKS.md) |

---

## Responsible Cluster Use

- **Always profile before scaling.** Run one step locally with `torch.profiler` before launching a cluster job. Most bugs are visible at small scale.
- **Use short-lived jobs.** Prefer many small experiments over one giant run. Set a `max_steps` budget before you launch.
- **Log everything.** Use Weights & Biases (free tier) or TensorBoard. A run you cannot reproduce is a run that did not happen.
- **Checkpoint frequently.** H100 jobs can be preempted. Save every N steps.
- **Share interesting runs.** If you find something surprising on the cluster, bring it to the next session. That is the whole point.

---

## Suggested Cluster Experiments

- **Week 9:** Run a proper FlashAttention vs standard attention benchmark at sequence lengths 512 / 1024 / 4096 / 8192. Plot IO-bound vs compute-bound crossover.
- **Week 10:** Train the same model in FP32, BF16, and (if stable) FP16. Compare loss curves, training time, and final perplexity. The H100's BF16 Tensor Core throughput makes this genuinely instructive.
- **Week 10:** Run a 2-GPU DDP job. Measure linear scaling.
- **Week 12:** Scale your model up one order of magnitude (e.g. ~10M to ~100M parameters) and train on a larger dataset slice. See how your architecture decisions hold up.

---

## Job Submission Template

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
# if using uv:
uv run python implementations/your-name/week09_flashattn_benchmark.py \
  --batch_size 8 \
  --seq_len 4096 \
  --max_steps 500
```

Save your job scripts in `implementations/your-name/cluster/` so others can learn from them.

---

## Logging Runs

Use Weights & Biases (free tier) or TensorBoard:

```python
import wandb
wandb.init(project="zdf-research", name="alice-week09-flashattn")
wandb.log({"train/loss": loss, "train/step": step})
```

Set your W&B entity to the group organisation so runs are visible to everyone.

---

## Cheap GPU Alternatives

For personal tinkering outside group sessions, **Lambda Labs** or **Vast.ai** (RTX 3090 / 4090, ~$0.30–0.50/hr) is usually sufficient. Keep the H100s for things the group is doing together.

---

## If You Accidentally Launch Something Expensive

Kill it. `scancel <job_id>`. No shame. The shame is in letting it run for six hours out of sunk-cost reasoning.
