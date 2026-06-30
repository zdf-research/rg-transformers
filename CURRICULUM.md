# 13-Week Curriculum: Build nanoGPT from Scratch

> Every week, the best implementation on the leaderboard becomes the group project's canonical module.

---

## At a Glance

| Week | Phase | Topic | Coding Milestone | Presenter | Week Metric | Winner |
| :--: | :---- | :---- | :--------------- | :-------- | :---------- | :----- |
| 1 | Build | [Embeddings & Tokenization](#week-1--embeddings--tokenization) | Character-level tokenizer + embedding lookup | TBD | Compression ratio, encode/decode parity | TBD |
| 2 | Build | [Gradients, Autograd & Why Recurrence Breaks](#week-2--gradients-autograd--why-recurrence-breaks) | MLP language model; inspect gradients with `gradcheck` | TBD | Gradient norm health, loss curve shape | TBD |
| 3 | Build | [Scaled Dot-Product Attention](#week-3--scaled-dot-product-attention) | Single-head causal self-attention from scratch | TBD | Numerical parity with `F.scaled_dot_product_attention` | TBD |
| 4 | Build | [Multi-Head Attention & Transformer Block](#week-4--multi-head-attention--the-transformer-block) | Multi-head attention + LayerNorm; stack blocks | TBD | Training loss on TinyShakespeare | TBD |
| 5 | Build | [Feed-Forward Networks & Activations](#week-5--feed-forward-networks--activations) | Parameterised FFN (ReLU/GELU/SwiGLU) | TBD | Loss vs parameter parity, forward speed | TBD |
| 6 | Build | [Positional Encodings & Full nanoGPT](#week-6--positional-encodings--full-nanogpt) | Full model: embed → blocks → LM head; train end-to-end | TBD | Validation loss, generation coherence (human eval) | TBD |
| | 🎉 | **Demo Day** | Each member demos text generation. Checkpoint the leaderboard. | | | |
| 7 | Optimise | [Rotary Position Embeddings (RoPE)](#week-7--rotary-position-embeddings-rope) | Replace absolute PE with RoPE; interleaved vs half-split | TBD | Extrapolation loss at 2× training context | TBD |
| 8 | Optimise | [GQA, KV Cache & torch.compile](#week-8--gqa-kv-cache--torchcompile) | MHA/GQA/MQA refactor; KV cache; graph-break audit | TBD | Generation tokens/sec, memory footprint | TBD |
| 9 | Optimise | [FlashAttention](#week-9--flashattention) | Tiled online softmax simulation; benchmark vs standard | TBD | Speedup factor, max batch size before OOM | TBD |
| 10 | Optimise | [Mixed Precision Training](#week-10--mixed-precision-training) | BF16 + grad clipping; loss curve logging | TBD | Throughput, memory, convergence parity | TBD |
| 11 | Optimise | [Distributed Training & Cluster Infra](#week-11--distributed-training--cluster-infra) | DDP on the cluster; SLURM, job submission, scaling laws | TBD | DDP scaling efficiency, multi-GPU throughput | TBD |
| 12 | Optimise | [Generation Strategies & Group Project Merge](#week-12--generation-strategies--group-project-merge) | Nucleus sampling, temperature; merge weekly winners | TBD | Perplexity, sample quality (group vote) | TBD |
| 13 | 🎉 | [Final Integration, Demo Day & Aperitivo](#week-13--final-integration-demo-day--aperitivo) | Final group model, live demos, celebration | TBD | Final group score + demo quality | TBD |

---

## Pre-Work (Before Week 1)

Watch **Andrej Karpathy — "Let's build GPT: from scratch, in code, spelled out"**
- Link: https://www.youtube.com/watch?v=kCc8FmEb1nY (~2 hours)
- This is mandatory. You do not need to understand every line yet, but you should be able to sketch the forward pass on a napkin by Week 1.

Optional gentler on-ramp: Karpathy's **makemore** series  
https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

---

## Phase 1 — Build It

---

### Week 1 — Embeddings & Tokenization

**The question:** How does a model represent meaning as a vector?

**Core references:**
- Karpathy, _makemore Part 1_ — bigram model and the embedding lookup table (video, ~1 hr): https://youtu.be/PaCmpygFfXo
- Jurafsky & Martin, _Speech and Language Processing_, Ch.s 2 - 5 — Vector Semantics and Embeddings (free): https://web.stanford.edu/~jurafsky/slp3/

**Deeper / optional:**
- Mikolov et al. (2013), _Efficient Estimation of Word Representations in Vector Space_: https://arxiv.org/abs/1301.3781
- CS224n Lecture 1 slides (Stanford): http://web.stanford.edu/class/cs224n/

**Code milestone:** Build a character-level tokenizer and embedding lookup layer. Then build a BPE tokenizer (GPT-2 style, ~8k vocab) and confirm your embedding layer handles the larger vocabulary.

> **Note:** The shared BPE tokenizer will be the standard for all leaderboard evaluation from Week 4 onwards. Character-level and BPE losses are not comparable, so we fix the token space early.

**Week metric:**
- Compression ratio (bytes after BPE encode / raw bytes)
- Encode/decode parity: does `decode(encode(text)) == text` hold on a held-out corpus?
- Embedding lookup speed (forward pass time on a batch)

**Discussion anchor:** Why a lookup table rather than one-hot vectors fed into a linear layer? Are they mathematically equivalent? Why does switching from character-level to BPE change the effective context window so dramatically?

**Extra credit:** Implement byte-fallback BPE so the tokenizer never produces `<unk>`.

---

### Week 2 — Gradients, Autograd & Why Recurrence Breaks

**The question:** What does backpropagation actually look like through a sequence model, and why does it break for long sequences?

**Core references:**
- Karpathy, _makemore Part 3_ — MLP language model and gradient flow (video): https://youtu.be/TCH_1BHY58I
- Karpathy, _makemore Part 4_ — deeper MLP, BatchNorm, initialisation (video): https://youtu.be/q8SA3rM6ckI
- Olah, _Understanding LSTM Networks_ (blog, read for intuition): https://colah.github.io/posts/2015-08-Understanding-LSTMs/

**Deeper / optional:**
- Hochreiter & Schmidhuber (1997), _Long Short-Term Memory_ — skim for how the cell state was designed
- d2l.ai, Ch. 9 — Recurrent Neural Networks: https://d2l.ai/chapter_recurrent-neural-networks/

**Code milestone:** Implement a 2–3 layer MLP language model. Use PyTorch autograd, but **inspect the gradients** manually: print `.grad` shapes, run `torch.autograd.gradcheck` on a single linear layer + loss, and plot gradient norms layer by layer. Observe how they shrink with depth — this is the vanishing gradient problem, made concrete.

> We do **not** write `loss.backward()` by hand this week. That is the Extra Credit. The goal is to understand *what* autograd is computing, not to replace it yet.

**Week metric:**
- Gradient norm ratio (first layer grad norm / last layer grad norm) — healthy networks keep this near 1
- Training loss curve shape on TinyShakespeare after 1000 steps
- `gradcheck` pass/fail on your modules

**Discussion anchor:** What is the sequential bottleneck in RNNs, and why can’t you parallelise training across the sequence dimension? How does attention sidestep this? (Save the full answer for Week 3 — plant the question here.)

**Extra credit:** Pick one weight matrix and compute its gradient by hand using the chain rule. Compare your result to autograd’s `.grad`. If they match within `1e-6`, you have understood backprop.

---

### Week 3 — Scaled Dot-Product Attention

**The question:** How does attention allow every token to directly query every other token, and why does it need a scaling factor?

**Core references:**
- Karpathy, _Let's build GPT_ — re-watch the attention section with fresh eyes (from ~45 min)
- Vaswani et al. (2017), _Attention Is All You Need_ — Sections 1–4 only: https://arxiv.org/abs/1706.03762
- d2l.ai, Ch. 11.3 — Attention Scoring Functions: https://d2l.ai/chapter_attention-mechanisms-and-transformers/

**Deeper / optional:**
- Bahdanau et al. (2015), _Neural Machine Translation by Jointly Learning to Align and Translate_: https://arxiv.org/abs/1409.0473

**Code milestone:** Single-head scaled dot-product self-attention with causal (lower-triangular) masking. Raw matrix operations only. Verify your output matches `torch.nn.functional.scaled_dot_product_attention` on a random input to within `1e-5`.

**Week metric:**
- Numerical parity with PyTorch reference (max absolute difference on random tensors)
- Max sequence length before OOM on a reference GPU (batch size 1)
- Forward pass latency at sequence lengths 256 / 512 / 1024

**Discussion anchor:** Derive the variance of the raw dot product $q \cdot k$ when components are i.i.d. $\mathcal{N}(0,1)$. Show it equals $d_k$, and therefore that dividing by $\sqrt{d_k}$ restores unit variance. Then: does this i.i.d. assumption hold after training? What saves us when it breaks?

**Extra credit:** Implement the attention pattern as a heatmap visualisation on a short sentence. What do you notice about the diagonal and the early tokens?

---

### Week 4 — Multi-Head Attention & the Transformer Block

**The question:** What does each head learn, and how do you stack these blocks stably?

**Core references:**
- Vaswani et al. (2017), _Attention Is All You Need_ — full paper now: https://arxiv.org/abs/1706.03762
- Ba et al. (2016), _Layer Normalization_: https://arxiv.org/abs/1607.06450
- d2l.ai, Ch. 11.5–11.7: https://d2l.ai/chapter_attention-mechanisms-and-transformers/

**Deeper / optional:**
- Xiong et al. (2020), _On Layer Normalization in the Transformer Architecture_ — Pre-LN vs Post-LN: https://arxiv.org/abs/2002.04745
- Zhang et al. (2019), _Root Mean Square Layer Normalization_ (RMSNorm): https://arxiv.org/abs/1910.07467

**Code milestone:** Full multi-head attention module. Add LayerNorm (pre-norm style). Stack $N$ transformer blocks. You now have a skeleton GPT.

**Week metric:**
- Training loss on TinyShakespeare after a fixed number of steps (e.g., 5000)
- Parameter count
- Wall-clock time per 1000 steps

**Discussion anchor:** Pre-norm vs post-norm — which trains more stably and why? What does each attention head specialise in?

**Extra credit:** Visualise attention weights for each head on the same sentence. Do different heads attend to different syntactic patterns?

---

### Week 5 — Feed-Forward Networks & Activations

**The question:** What is the FFN block actually doing, and why do modern models use gated activations?

**Core references:**
- Vaswani et al. (2017) — re-read the FFN section closely
- Shazeer (2020), _GLU Variants Improve Transformer_: https://arxiv.org/abs/2002.05202
- d2l.ai, Ch. 11.4: https://d2l.ai/chapter_attention-mechanisms-and-transformers/

**Deeper / optional:**
- Hendrycks & Gimpel (2016), _Gaussian Error Linear Units (GELUs)_: https://arxiv.org/abs/1606.08415
- Ramachandran et al. (2017), _Searching for Activation Functions_: https://arxiv.org/abs/1710.05941

**Code milestone:** Parameterise your FFN block to support vanilla ReLU, GELU, and SwiGLU. When switching to SwiGLU (3 weight matrices), scale the hidden dimension by $2/3$ to maintain approximate parameter parity with the baseline FFN.

> **Caveat:** Round scaled hidden dimensions up to the nearest multiple of 64. Hardware kernels in later weeks require this. A hidden dim of 683 will silently cause shape mismatches in compiled kernels.

**Week metric:**
- Validation loss on TinyShakespeare (fixed steps, fixed seed)
- Parameter count after scaling
- Forward pass throughput (tokens/sec)

**Discussion anchor:** Why does gating help? Interpret the gate branch as a learned binary mask. Is the $2/3$ scaling factor principled or heuristic?

**Extra credit:** Ablate the FFN entirely — replace it with a single linear layer. How much does loss degrade?

---

### Week 6 — Positional Encodings & Full nanoGPT

**The question:** Self-attention is permutation-invariant — so how do we tell the model where each token is?

**Core references:**
- Vaswani et al. (2017) — positional encoding section
- Karpathy, _Let's build GPT_ — positional embedding table implementation
- d2l.ai, Ch. 11.6: https://d2l.ai/chapter_attention-mechanisms-and-transformers/

**Deeper / optional:**
- Press et al. (2021), _ALiBi_: https://arxiv.org/abs/2108.12409
- Su et al. (2021), _RoFormer — preview for Week 7_: https://arxiv.org/abs/2104.09864

**Code milestone:** Add a learned absolute position embedding table. Wire everything together: embedding + positional encoding → $N$ transformer blocks → LM head. Train on TinyShakespeare. **This is your Phase 1 complete model.** Run the shared eval and post your validation loss to the leaderboard.

**Week metric:**
- Validation cross-entropy loss (the first true leaderboard metric)
- Human-evaluated generation coherence (group vote on sample quality)
- Training time to target loss

**Discussion anchor:** What are the limitations of learned absolute position embeddings? What happens at inference time if you exceed the training context length?

**Extra credit:** Implement sinusoidal (fixed) position encodings instead of learned. Do you observe a difference in convergence speed?

---

### 🎉 Demo Day — After Week 6

Each participant:

1. Shows their model generating text (Shakespeare, or any fun prompt)
2. Reports their leaderboard position and what they think drove their loss
3. Names one thing they would change in Phase 2

Checkpoint the leaderboard. Celebrate. Then continue.

---

## Phase 2 — Understand & Optimise It

---

### Week 7 — Rotary Position Embeddings (RoPE)

**The question:** Can we encode position in the attention scores themselves, so it naturally generalises to longer sequences?

**Core references:**
- Su et al. (2021), _RoFormer: Enhanced Transformer with Rotary Position Embedding_: https://arxiv.org/abs/2104.09864
- EleutherAI blog, _Rotary Embeddings: A Relative Revolution_: https://blog.eleuther.ai/rotary-embeddings/

**Deeper / optional:**
- Yang et al. (2024), _Gated Linear Attention Transformers with Hardware-Efficient Training_: https://arxiv.org/abs/2312.06635
- Lilian Weng, _Understanding Positional Encoding in Transformers_: https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/

**Code milestone:** Replace your absolute position table with RoPE. Implement both interleaved and half-split memory layouts. Verify both converge to the same loss region. Note any runtime difference.

**Week metric:**
- Validation loss at training context length
- Validation loss at **2× training context length** (extrapolation test)
- Forward pass latency difference: RoPE vs absolute PE

**Discussion anchor:** Derive the inner product $\langle R_m q, R_n k \rangle$ and show it depends only on $(m - n)$. Why does this give "relative" position for free?

**Extra credit:** Implement RoPE with xPos scaling (used in extensions like Yarn). Does it improve extrapolation beyond 2×?

---

### Week 8 — GQA, KV Cache & torch.compile

**The question:** During inference, why does memory bandwidth become the bottleneck, and how does GQA fix it?

**Core references:**
- Ainslie et al. (2023), _GQA: Training Generalised Multi-Query Transformer Models from Multi-Head Checkpoints_: https://arxiv.org/abs/2305.13245
- Shazeer (2019), _Fast Transformer Decoding: One Write-Head is All You Need_ (MQA): https://arxiv.org/abs/1911.02150

**Deeper / optional:**
- Pope et al. (2022), _Efficiently Scaling Transformer Inference_: https://arxiv.org/abs/2211.05100

**Code milestone:** Refactor multi-head attention to support MHA / GQA / MQA via a single `num_kv_heads` parameter. Implement a KV cache for autoregressive generation. Then run a **graph-break audit**:

```python
import torch._dynamo
torch._dynamo.config.verbose = True
compiled_model = torch.compile(model)
# run one forward pass and inspect logs for "graph break" warnings
```

Fix any breaks caused by Python control flow, lists-instead-of-tensors in the cache, or `.item()` calls before moving to Week 9.

**Week metric:**
- Generation throughput (tokens/sec) with and without KV cache
- Memory usage at context length 2048 for MHA vs GQA-4 vs MQA
- Number of graph breaks found and fixed

**Discussion anchor:** Calculate the KV cache size for your model at context length 2048 under MHA vs GQA-4 vs MQA. How does this translate to max batch size on a 24 GB GPU?

**Extra credit:** Implement speculative decoding: use a 2-layer draft model to propose tokens, verified in parallel by your main model. Measure end-to-end speedup.

---

### Week 9 — FlashAttention

**The question:** Why is standard attention memory-bound rather than compute-bound, and how does tiling fix it?

**Core references:**
- Dao et al. (2022), _FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness_: https://arxiv.org/abs/2205.14135
- Dao et al. (2023), _FlashAttention-2: Faster Attention with Better Parallelism_: https://arxiv.org/abs/2307.08691
- Horace He, _Making Deep Learning Go Brrrr From First Principles_: https://horace.io/brrr_intro.html

**Deeper / optional:**
- Dao (2024), _FlashAttention-3_: https://arxiv.org/abs/2407.08608
- CUDA Mode lectures: https://www.youtube.com/@cudamode

**Code milestone:** Implement a pure-PyTorch **simulation** of tiled online softmax (you do not need to write CUDA). Walk through the IO complexity: count HBM reads/writes for standard attention vs tiled attention. Then switch your model to `torch.nn.functional.scaled_dot_product_attention` (dispatches to FlashAttention on CUDA) and benchmark.

**Week metric:**
- Speedup factor: FlashAttention vs your Week 3 naive attention
- Maximum batch size before OOM at sequence length 2048
- Wall-clock time at sequence lengths 512 / 1024 / 4096 / 8192

**Discussion anchor:** Derive the online softmax update rule. Why is it numerically identical to standard softmax? What is the tradeoff between tile size and SRAM usage?

**Extra credit:** Profile your model with the PyTorch profiler. Where does time go *after* FlashAttention removes attention as the bottleneck?

---

### Week 10 — Mixed Precision Training

**The question:** How do you train stably in half precision?

**Core references:**
- Micikevicius et al. (2018), _Mixed Precision Training_: https://arxiv.org/abs/1710.03740
- Karpathy, _Let's reproduce GPT-2_ — mixed precision and gradient scaling sections
- d2l.ai, Ch. 19 — Computational Performance: https://d2l.ai/chapter_computational-performance/

**Deeper / optional:**
- Dettmers et al. (2022), _LLM.int8()_ — quantisation context: https://arxiv.org/abs/2208.07339
- The BF16 vs FP16 explainer in the original LLaMA technical report appendix

**Code milestone:** Switch your training loop to BF16 mixed precision using `torch.autocast`. Add gradient clipping. Log loss curves and gradient norms. Try reproducing a small known result: train on TinyShakespeare to a target validation loss and compare your curve shape to Karpathy's reference.

**Week metric:**
- Training throughput (tokens/sec) in FP32 vs BF16
- Memory consumption reduction in BF16
- Convergence parity: does BF16 reach the same validation loss as FP32?

**Discussion anchor:** Why does BF16 train more stably than FP16 out of the box? (Hint: exponent bits.) When would you prefer FP16? What is dynamic loss scaling and when do you need it?

**Extra credit:** Train the same model in FP32, BF16, and (if you can stably) FP16. Plot all three loss curves on the same axes.

---

### Week 11 — Distributed Training & Cluster Infra

**The question:** What does it take to run on multiple GPUs, and how do you actually submit jobs to the cluster?

**Core references:**
- PyTorch DDP tutorial: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- Rajbhandari et al. (2020), _ZeRO: Memory Optimizations Toward Training Trillion Parameter Models_: https://arxiv.org/abs/1910.02054
- Shoeybi et al. (2019), _Megatron-LM_: https://arxiv.org/abs/1909.08053

**Deeper / optional:**
- PyTorch FSDP documentation: https://pytorch.org/docs/stable/fsdp.html

**Code milestone:** Submit a **minimal 2-GPU DDP job** to the cluster. Measure linear scaling. Walk through the SLURM template, environment setup with `uv`, and what `torchrun` or `mpirun` actually does. This is the infrastructure week — expect YAML, not just Python.

**Week metric:**
- DDP scaling efficiency (2-GPU throughput / 1-GPU throughput)
- Job submission time (how long from `sbatch` to first logged step?)
- Max model size that fits on 2× H100s with DDP

**Discussion anchor:** Why does DDP require an `all_reduce` on every backward pass? When would you use FSDP instead of DDP? What does ZeRO-1 actually do, and why is it different from DDP?

**Extra credit:** Profile a ZeRO-1 or FSDP setup. How much memory does offloading optimizer states to CPU save?

---

### Week 12 — Generation Strategies & Group Project Merge

**The question:** How do you know if your model is actually good, and how do we merge 13 weeks of work into one system?

**Core references:**
- Jurafsky & Martin, _SLP3_, Ch. 3 — perplexity definition
- Liang et al. (2022), _Holistic Evaluation of Language Models (HELM)_: https://arxiv.org/abs/2211.09110
- Holtzman et al. (2020), _The Curious Case of Neural Text Degeneration_ (nucleus sampling): https://arxiv.org/abs/1904.09751

**Deeper / optional:**
- Srivastava et al. (2022), _BIG-Bench_: https://arxiv.org/abs/2206.04615
- Biderman et al. (2023), _Pythia_: https://arxiv.org/abs/2304.01373

**Code milestone:** Implement nucleus (top-p) sampling and temperature scaling. Run the final leaderboard eval. Then begin **merging the weekly winners** into a single `group_project/` model: each week's best tokenizer, embedding, attention, FFN, and so on.

**Week metric:**
- Perplexity on the fixed held-out split
- Generation sample quality (group anonymous vote, 1–5 scale)
- Diversity of outputs (unique n-gram ratio)

**Discussion anchor:** Perplexity is easy to compute but what does it miss? When does a lower-perplexity model generate *worse* text? How do temperature and top-p interact?

**Extra credit:** Implement repetition penalty or frequency-based sampling. Does it improve perceived quality?

---

### Week 13 — Final Integration, Demo Day & Aperitivo

**The question:** What did we build?

**Code milestone:** Final integration. The `group_project/` directory contains the group's canonical transformer, assembled from weekly winners. Each module is attributed to its author. Run the full evaluation suite and final leaderboard snapshot.

**Week metric:**
- Final group model validation loss
- Final group model parameter count
- Demo quality (group vote + any invited guests)

**Final presentations:** Each participant:

1. Architecture overview — what choices did you make and why?
2. Training curve — what did you learn from it?
3. A live generation demo
4. Final leaderboard position and post-mortem

Then: **aperitivo**. You have earned it.

Optionally: write up a shared group blog post or technical report. This is the CV artefact.

---

## Quick Reference: Resources by Type

### Videos

| Resource | Link | When |
| :------- | :--- | :--- |
| Karpathy — makemore series | https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ | Pre-work / Weeks 1–2 |
| Karpathy — Let's build GPT | https://youtu.be/kCc8FmEb1nY | **Pre-work (mandatory)** |
| Karpathy — Let's reproduce GPT-2 | https://youtu.be/l8pRSuU81PU | Weeks 10–11 |
| CUDA Mode lectures | https://www.youtube.com/@cudamode | Weeks 9–10 (optional) |

### Textbooks (free online)

| Resource | Link | When |
| :------- | :--- | :--- |
| d2l.ai | https://d2l.ai | Throughout |
| Jurafsky & Martin, SLP3 | https://web.stanford.edu/~jurafsky/slp3/ | Weeks 1, 11 |

### Course Materials

| Resource | Link | When |
| :------- | :--- | :--- |
| Stanford CS224n | http://web.stanford.edu/class/cs224n/ | Supplementary |

### Essential Papers

| Paper | Link | Week |
| :---- | :--- | :--- |
| Vaswani et al. (2017) — Attention Is All You Need | https://arxiv.org/abs/1706.03762 | 3–4 |
| Ba et al. (2016) — Layer Normalization | https://arxiv.org/abs/1607.06450 | 4 |
| Shazeer (2020) — GLU Variants | https://arxiv.org/abs/2002.05202 | 5 |
| Su et al. (2021) — RoFormer / RoPE | https://arxiv.org/abs/2104.09864 | 7 |
| Ainslie et al. (2023) — GQA | https://arxiv.org/abs/2305.13245 | 8 |
| Dao et al. (2022) — FlashAttention | https://arxiv.org/abs/2205.14135 | 9 |
| Dao et al. (2023) — FlashAttention-2 | https://arxiv.org/abs/2307.08691 | 9 |
| Micikevicius et al. (2018) — Mixed Precision | https://arxiv.org/abs/1710.03740 | 10 |
| Rajbhandari et al. (2020) — ZeRO | https://arxiv.org/abs/1910.02054 | 11 |
| Shoeybi et al. (2019) — Megatron-LM | https://arxiv.org/abs/1909.08053 | 11 |
