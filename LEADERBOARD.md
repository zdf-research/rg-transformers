# Leaderboard

> Updated after each session. See [CURRICULUM.md](CURRICULUM.md) for per-week metrics.

## Phase 1 — Build It

| Week | Name | Primary Metric | Secondary Metric | Notes |
| :--: | :--- | :------------- | :--------------- | :---- |
| 1 | — | — | — | Awaiting first submissions |
| 2 | — | — | — | |
| 3 | — | — | — | |
| 4 | — | — | — | |
| 5 | — | — | — | |
| 6 | — | — | — | Demo Day checkpoint |

### Week 1 — Tokenizer & Embedding
**Metric:** Compression ratio, encode/decode parity, lookup speed

| Name | Compression Ratio | Parity Pass | Lookup Speed (ms/batch) | Notes |
| :--- | :---------------- | :---------- | :---------------------- | :---- |
| — | — | — | — | |

### Week 2 — Gradients & Autograd
**Metric:** Gradient norm ratio, loss curve shape, gradcheck pass

| Name | Grad Norm Ratio | Loss @ 1k steps | Gradcheck | Notes |
| :--- | :-------------- | :-------------- | :-------- | :---- |
| — | — | — | — | |

### Week 3 — Attention
**Metric:** Numerical parity with reference, max seq len before OOM, latency

| Name | Max Diff | Max Seq Len | Latency 512 (ms) | Notes |
| :--- | :------- | :---------- | :--------------- | :---- |
| — | — | — | — | |

### Week 4 — Transformer Block
**Metric:** Training loss on TinyShakespeare, param count, wall time

| Name | Val Loss | Params | Time/1k steps | Notes |
| :--- | :------- | :----- | :------------ | :---- |
| — | — | — | — | |

### Week 5 — FFN
**Metric:** Validation loss, param parity, forward throughput

| Name | Val Loss | Params | Throughput | Notes |
| :--- | :------- | :----- | :--------- | :---- |
| — | — | — | — | |

### Week 6 — Full nanoGPT
**Metric:** Validation loss, generation coherence (human eval), train time

| Name | Val Loss | Human Eval (1–5) | Train Time | Notes |
| :--- | :------- | :--------------- | :--------- | :---- |
| — | — | — | — | |

---

## Phase 2 — Optimise It

| Week | Name | Primary Metric | Secondary Metric | Notes |
| :--: | :--- | :------------- | :--------------- | :---- |
| 7 | — | — | — | |
| 8 | — | — | — | |
| 9 | — | — | — | |
| 10 | — | — | — | |
| 11 | — | — | — | |
| 12 | — | — | — | Final Demo Day |

### Week 7 — RoPE
**Metric:** Extrapolation loss at 2× context, latency

| Name | Val Loss (train ctx) | Val Loss (2× ctx) | Latency | Notes |
| :--- | :------------------- | :---------------- | :------ | :---- |
| — | — | — | — | |

### Week 8 — GQA & KV Cache
**Metric:** Generation tokens/sec, memory at 2048, graph breaks fixed

| Name | Tok/sec | Memory (GB) | Graph Breaks | Notes |
| :--- | :------ | :---------- | :----------- | :---- |
| — | — | — | — | |

### Week 9 — FlashAttention
**Metric:** Speedup vs naive, max batch size, latency by seq len

| Name | Speedup | Max Batch (seq 2048) | Latency 4096 (ms) | Notes |
| :--- | :------ | :------------------- | :---------------- | :---- |
| — | — | — | — | |

### Week 10 — Mixed Precision & DDP
**Metric:** Throughput, memory, convergence parity, DDP scaling

| Name | Tok/sec (BF16) | Memory | Convergence Parity | DDP Efficiency | Notes |
| :--- | :------------- | :----- | :----------------- | :------------- | :---- |
| — | — | — | — | — | |

### Week 11 — Generation & Merge
**Metric:** Perplexity, sample quality, diversity

| Name | Perplexity | Sample Quality | Diversity | Notes |
| :--- | :--------- | :------------- | :-------- | :---- |
| — | — | — | — | |

### Week 12 — Generation & Merge
**Metric:** Perplexity, sample quality, diversity

| Name | Perplexity | Sample Quality | Diversity | Notes |
| :--- | :--------- | :------------- | :-------- | :---- |
| — | — | — | — | |

### Week 13 — Final Integration
**Metric:** Group model loss, parameter count, demo quality

| Name | Val Loss | Params | Demo Score | Notes |
| :--- | :------- | :----- | :--------- | :---- |
| — | — | — | — | |

---

## Group Project — Weekly Winners

| Week | Winning Module | Author | PR to `group_project/` |
| :--- | :------------- | :----- | :------------------- |
| 1 | Tokenizer & Embedding | TBD | — |
| 2 | MLP + Gradient Inspector | TBD | — |
| 3 | Attention | TBD | — |
| 4 | Transformer Block | TBD | — |
| 5 | FFN / SwiGLU | TBD | — |
| 6 | Full nanoGPT | TBD | — |
| 7 | RoPE | TBD | — |
| 8 | GQA + KV Cache | TBD | — |
| 9 | FlashAttention Integration | TBD | — |
| 10 | Mixed Precision Training | TBD | — |
| 11 | Distributed Training | TBD | — |
| 12 | Generation & Sampling | TBD | — |
| 13 | Final Integration | TBD | — |
