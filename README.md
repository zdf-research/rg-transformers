# Language Modelling & Transformers — Reading & Build Group

> _Because at some point, someone has to build the thing that makes the dashboards obsolete._

---

## Why This Group Exists

Let's be honest about how a significant fraction of our working week goes. There is the SQL query that a stakeholder could have written themselves in ten minutes but instead becomes a three-email thread and a meeting. There is the Power BI report that must be rebuilt from scratch because someone changed a column name. There is the data extraction that is, conceptually, a `SELECT *`, dressed up as a project. There is the "quick alignment call" that is neither quick nor aligning anyone on anything. And there is, somewhere underneath all of it, the reason most of us got into this field in the first place: because machine learning is genuinely one of the most interesting things humans have ever built, and we would quite like to understand how it actually works.

This group is the antidote. One session a week, no stakeholders, no Confluence pages, no one asking if we can "just add a filter for region." Just a small group of people who want to go from first principles to a production-quality autoregressive transformer — and understand every line of the way.

The technical pitch: most of us have _used_ transformers. Far fewer can explain, from scratch, _why_ the scaling factor in attention is $1/\sqrt{d_k}$, what actually happens in GPU memory during a forward pass, or how RoPE encodes relative position without ever storing a position matrix. Tutorials get you to the API. This group gets you to the internals.

The goal is simple: over 12 weeks, working together, every participant builds a complete, well-understood, hardware-aware autoregressive language model — from the embedding table to the generation loop. Every line of code is written, reviewed, and profiled by the group. Every design decision is traced back to a paper.

By the end, you will have:

- A working nanoGPT-style transformer, built from raw matrix operations
- Deep familiarity with the papers that define the modern LLM stack
- A shared GitHub repository with your code, your presentation slides, and your model checkpoints
- A leaderboard showing validation loss across implementations — fair comparison, fixed dataset
- A project you can talk about in depth in any technical interview
- And, perhaps most importantly, ninety minutes a week where nobody asks you to reformat a pivot table

---

## How Sessions Work

We meet **weekly**, for roughly **90 minutes**. Each session has two parts:

### Part 1 — Paper & Concept Presentation (45 min)

One participant presents the week's core material. This is not a summary reading — it is a teach-it-back session. The presenter is expected to:

- Derive the key equations on a whiteboard (or slides)
- Explain the _motivation_ for the design, not just the mechanics
- Flag one or two things they found confusing or that the paper does not explain well
- Field questions from the group

Slides go into the shared repo under `/presentations/weekNN-your-name/`.

### Part 2 — Code Review Lab (45 min)

Everyone pushes their implementation of the week's module before the session. During the lab:

- One or two people walk through their code
- The group discusses correctness, clarity, and efficiency
- We run a shared profiling script and log results to the leaderboard

Code goes into `/implementations/your-name/weekNN/`.

---

## Repository Structure

```
rg-transformers/
├── README.md                  ← this document
├── CONTRIBUTING.md            ← how to push code, use the cluster, update the leaderboard
├── LEADERBOARD.md             ← auto-updated validation loss table
├── common/                    ← shared utilities (protected — PR required to modify)
│   ├── tokenizer.py           ← shared BPE tokeniser
│   ├── data.py                ← dataset loading (TinyShakespeare, WikiText-103)
│   ├── eval.py                ← evaluation script — do not modify without group consensus
│   ├── train.py               ← reference training loop
│   └── requirements.txt       ← pinned dependencies
├── presentations/             ← weekly seminar slides, one folder per session
│   ├── week01-alice/
│   │   └── slides.pdf
│   └── week03-bob/
│       └── slides.pdf
└── implementations/           ← one folder per member, push freely
    ├── alice/
    │   ├── README.md
    │   ├── week01_embeddings.py
    │   ├── week03_attention.py
    │   └── ...
    └── bob/
        └── ...
```

**Ground rules:**

- `common/` is shared — changes go through a PR and a quick group review
- Your implementation folder is yours — commit freely, break things freely
- The leaderboard is updated by running `common/eval.py` against a fixed held-out split; do not change the eval script without group consensus

---

## The Leaderboard

The leaderboard lives in `LEADERBOARD.md` and tracks **validation loss (cross-entropy)** on a fixed held-out slice of WikiText-103, updated after each session's code lab. Secondary metrics (parameters, training time, tokens/sec on a reference GPU) are tracked from Week 7 onwards when models become more comparable.

The point of the leaderboard is not competition — it is accountability and debugging. If your loss is meaningfully worse than others on the same architecture, something is wrong, and the group helps you find it.

---

## Phase Structure

The 12 weeks are split into two phases, with a **mini demo day** between them.

| Phase                                  | Weeks        | Goal                                                                            |
| -------------------------------------- | ------------ | ------------------------------------------------------------------------------- |
| **Phase 1 — Build It**                 | 1 – 6        | A working character-level transformer, trained and generating text              |
| **↳ Demo Day**                         | After Week 6 | Each participant demos their model generating text. Checkpoint the leaderboard. |
| **Phase 2 — Understand & Optimise It** | 7 – 12       | RoPE, GQA, attention efficiency, evaluation, the full picture                   |

---

## Pre-Work (Before Week 1)

Before the first session, everyone watches:

> 🎬 **Andrej Karpathy — "Let's build GPT: from scratch, in code, spelled out"**  
> https://www.youtube.com/watch?v=kCc8FmEb1nY (~2 hours)

This is mandatory. It gives everyone the same mental model of the full system before we start dissecting individual parts. You do not need to understand every line — that is what the next 12 weeks are for. But you should be able to sketch the forward pass on a napkin by Week 1.

Optionally, also watch his earlier **makemore** series (character-level MLP and bigram models) if you want a gentler on-ramp:  
https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

---

## 12-Week Schedule

### Phase 1 — Build It

---

#### Week 1 — Vector Semantics & the Embedding Layer

**The question:** How does a model represent meaning as a vector?

**Core references (do these):**

- Karpathy, _makemore Part 1_ — bigram model and the embedding lookup table (video, ~1 hr): https://youtu.be/PaCmpygFfXo
- Jurafsky & Martin, _Speech and Language Processing_, Ch. 6 — Vector Semantics and Embeddings (free online): https://web.stanford.edu/~jurafsky/slp3/

**Deeper / optional:**

- Mikolov et al. (2013), _Efficient Estimation of Word Representations in Vector Space_ (Word2Vec paper): https://arxiv.org/abs/1301.3781
- CS224n Lecture 1 slides (Stanford): http://web.stanford.edu/class/cs224n/

**Code milestone:** A matrix-multiplication-based token embedding lookup layer. Build a character-level tokeniser first — it is the simplest possible case and makes the lookup table concept concrete. Then plug in the shared BPE tokeniser from `common/tokenizer.py` (GPT-2 style, ~8k vocab) and confirm your embedding layer handles the larger vocabulary. **The shared BPE tokeniser is the standard for all leaderboard evaluation from this point forward** — character-level and BPE losses are not comparable numbers on the same dataset, so fixing the token space now prevents a leaderboard mess later.

**Discussion anchor:** Why do we use a lookup table rather than one-hot vectors fed into a linear layer? Are these mathematically equivalent? What changes at scale? Why does switching from character-level to BPE change the effective context window so dramatically — 256 character tokens is roughly 50 words, but 256 BPE tokens is closer to 150–200 words?

---

#### Week 2 — Gradient Flow, Vanishing Gradients & Why Recurrence Breaks

**The question:** What does backpropagation actually look like through a sequence model, and why does it break for long sequences?

**Core references:**

- Karpathy, _makemore Part 3_ — MLP language model, manual backprop, gradient flow (video): https://youtu.be/TCH_1BHY58I
- Karpathy, _makemore Part 4_ — deeper MLP, BatchNorm, initialisation (video): https://youtu.be/q8SA3rM6ckI
- Olah, _Understanding LSTM Networks_ (blog post — read for intuition, not implementation): https://colah.github.io/posts/2015-08-Understanding-LSTMs/

**Deeper / optional:**

- Hochreiter & Schmidhuber (1997), _Long Short-Term Memory_ — skim the original paper to see how the cell state was designed to solve the gradient problem
- d2l.ai, Ch. 9 — Recurrent Neural Networks: https://d2l.ai/chapter_recurrent-neural-networks/
- CS224n Lectures 5 & 6

**Code milestone:** Implement a 2–3 layer MLP language model (following Karpathy's makemore Part 3) and write the backward pass **manually** — no `loss.backward()`. Compute gradients for each weight matrix by hand using the chain rule. Plot gradient norms layer by layer and observe how they shrink with depth. This is the vanishing gradient problem, made concrete without the algebraic nightmare of LSTM cell derivatives.

> **Why not LSTM BPTT?** Manual backpropagation through an LSTM requires simultaneously tracking gradients through four gated branches and two state vectors. It is a worthy exercise eventually, but as Week 2 of a group that needs momentum, it is a trap. The MLP gives you the same core insight — gradient flow degrades with depth and sequence length — without derailing the first month. Once you understand transformers deeply, the LSTM derivation becomes an interesting afternoon, not a wall.

**Discussion anchor:** What is the sequential bottleneck in RNNs, and why can't you parallelise training across the sequence dimension? How does the attention mechanism sidestep this? (Save the full answer for Week 3 — just plant the question here.)

---

#### Week 3 — Scaled Dot-Product Attention

**The question:** How does attention allow every token to directly query every other token, and why does it need a scaling factor?

**Core references:**

- Karpathy, _Let's build GPT_ — re-watch the attention section with fresh eyes (from ~45 min mark)
- Vaswani et al. (2017), _Attention Is All You Need_ — Sections 1–4 only to start: https://arxiv.org/abs/1706.03762
- d2l.ai, Ch. 11.3 — Attention Scoring Functions: https://d2l.ai/chapter_attention-mechanisms-and-transformers/

**Deeper / optional:**

- Bahdanau et al. (2015), _Neural Machine Translation by Jointly Learning to Align and Translate_ — the original attention paper, good for historical context
- The full Vaswani et al. paper

**Code milestone:** Single-head scaled dot-product self-attention with causal (lower-triangular) masking. No libraries — raw matrix operations only. Verify that your output matches `torch.nn.functional.scaled_dot_product_attention` on a random input.

**Discussion anchor:** Derive the variance of the raw dot product $q \cdot k$ when components are i.i.d. $\mathcal{N}(0,1)$ — show it equals $d_k$, and therefore that dividing by $\sqrt{d_k}$ restores unit variance. Present the derivation to the group. Then ask the harder follow-up: **this argument assumes i.i.d. components, which holds at initialisation but breaks during training as features become correlated. Does softmax still saturate post-training even with the scaling factor?** (It can — which is precisely why LayerNorm inside or before the attention block is empirically mandatory, not optional. The scaling factor buys you stability at init; the norm buys you stability throughout training.)

---

#### Week 4 — Multi-Head Attention & the Transformer Block

**The question:** What does each head learn, and how do you stack these blocks stably?

**Core references:**

- Vaswani et al. (2017), _Attention Is All You Need_ — full paper now: https://arxiv.org/abs/1706.03762
- Ba et al. (2016), _Layer Normalization_: https://arxiv.org/abs/1607.06450
- d2l.ai, Ch. 11.5–11.7: https://d2l.ai/chapter_attention-mechanisms-and-transformers/

**Deeper / optional:**

- Xiong et al. (2020), _On Layer Normalization in the Transformer Architecture_ — Pre-LN vs Post-LN analysis: https://arxiv.org/abs/2002.04745
- Zhang et al. (2019), _Root Mean Square Layer Normalization_ (RMSNorm): https://arxiv.org/abs/1910.07467

**Code milestone:** Full multi-head attention module. Add LayerNorm (pre-norm style). Stack N transformer blocks. You now have a skeleton GPT.

**Discussion anchor:** Pre-norm vs post-norm — which trains more stably and why? What does each attention head specialise in? (Try visualising attention weights on a short sequence.)

---

#### Week 5 — The Feed-Forward Sublayer & Activation Functions

**The question:** What is the FFN block actually doing, and why do modern models use gated activations?

**Core references:**

- Vaswani et al. (2017) — re-read the FFN section closely
- Shazeer (2020), _GLU Variants Improve Transformer_: https://arxiv.org/abs/2002.05202
- d2l.ai, Ch. 11.4: https://d2l.ai/chapter_attention-mechanisms-and-transformers/

**Deeper / optional:**

- Hendrycks & Gimpel (2016), _Gaussian Error Linear Units (GELUs)_: https://arxiv.org/abs/1606.08415
- Ramachandran et al. (2017), _Searching for Activation Functions_ (Swish/SiLU): https://arxiv.org/abs/1710.05941

**Code milestone:** Parameterise your FFN block to support vanilla ReLU, GELU, and SwiGLU. When switching to SwiGLU (which needs 3 weight matrices instead of 2), scale the hidden dimension by 2/3 to maintain approximate parameter parity with the baseline FFN. Two important caveats: first, the 2/3 factor only gives exact parity if your baseline used $d_{\text{ff}} = 4 \times d_{\text{model}}$ — verify this for your config. Second, hardware kernels in Weeks 9–10 require dimensions to be multiples of 64. Always round your scaled hidden dimension up to the nearest multiple of 64 after applying the factor (e.g. `int(d_ff * 2/3 / 64) * 64`). A hidden dimension of 683 will silently cause shape mismatches in compiled kernels two months from now. Fix it once, here.

**Discussion anchor:** Why does gating help? Interpret the gate branch as a learned binary mask — what does it mean for the model to "select" which features to pass through? Is the 2/3 scaling factor a principled choice or a heuristic?

---

#### Week 6 — Positional Encodings & the Full nanoGPT

**The question:** Self-attention is permutation-invariant — so how do we tell the model where each token is?

**Core references:**

- Vaswani et al. (2017) — the positional encoding section
- Karpathy, _Let's build GPT_ — the positional embedding table implementation
- d2l.ai, Ch. 11.6 — Self-Attention and Positional Encoding: https://d2l.ai/chapter_attention-mechanisms-and-transformers/

**Deeper / optional:**

- Press et al. (2021), _Train Short, Test Long: Attention with Linear Biases (ALiBi)_: https://arxiv.org/abs/2108.12409
- Su et al. (2021), _RoFormer: Enhanced Transformer with Rotary Position Embedding_ — preview for Week 7: https://arxiv.org/abs/2104.09864

**Code milestone:** Add a learned absolute position embedding table. Wire everything together: embedding + positional encoding → N transformer blocks → LM head. Train on TinyShakespeare. **This is your Phase 1 complete model.** Run `common/eval.py` and post your validation loss to the leaderboard.

**Discussion anchor:** What are the limitations of learned absolute position embeddings? What happens at inference time if you exceed the training context length?

---

### 🎉 Demo Day — After Week 6

Each participant:

1. Shows their model generating text (Shakespeare, or any fun prompt)
2. Reports their leaderboard position and what they think drove their loss
3. Names one thing they would change in Phase 2

Checkpoint the leaderboard. Celebrate. Then continue.

---

### Phase 2 — Understand & Optimise It

---

#### Week 7 — Rotary Position Embeddings (RoPE)

**The question:** Can we encode position in the attention scores themselves, so it naturally generalises to longer sequences?

**Core references:**

- Su et al. (2021), _RoFormer: Enhanced Transformer with Rotary Position Embedding_: https://arxiv.org/abs/2104.09864
- Karpathy's RoPE implementation notes (nanoGPT issues / discussions on GitHub)
- Blog: _Rotary Embeddings: A Relative Revolution_ (EleutherAI): https://blog.eleuther.ai/rotary-embeddings/

**Deeper / optional:**

- Yang et al. (2024), _Gated Linear Attention Transformers with Hardware-Efficient Training_: https://arxiv.org/abs/2312.06635
- Blog: _Understanding Positional Encoding in Transformers_ (Lilian Weng): https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/

**Code milestone:** Replace your absolute position table with RoPE. Implement both the interleaved and half-split memory layouts. **Do not expect identical numerical outputs** — the two layouts pair rotation frequencies to different index positions in the tensor, so they represent different geometric transformations of the same vector unless you explicitly permute the weight matrices or input states to match. What you should verify is that both are _mathematically equivalent in representational capacity_: run a small training loop with each layout and confirm that loss curves converge to the same region. Note any runtime difference between the two layouts (interleaved tends to be faster due to cache locality).

**Discussion anchor:** Derive the inner product $\langle R_m q, R_n k \rangle$ and show it depends only on $(m - n)$. Why does this give "relative" position for free?

---

#### Week 8 — Grouped-Query Attention & the KV Cache

**The question:** During inference, why does memory bandwidth become the bottleneck, and how does GQA fix it?

**Core references:**

- Ainslie et al. (2023), _GQA: Training Generalised Multi-Query Transformer Models from Multi-Head Checkpoints_: https://arxiv.org/abs/2305.13245
- Shazeer (2019), _Fast Transformer Decoding: One Write-Head is All You Need_ (MQA original): https://arxiv.org/abs/1911.02150

**Deeper / optional:**

- Pope et al. (2022), _Efficiently Scaling Transformer Inference_ (KV cache analysis): https://arxiv.org/abs/2211.05100

**Code milestone:** Refactor your multi-head attention to support MHA / GQA / MQA via a single `num_kv_heads` parameter. Implement a KV cache for autoregressive generation (store past K, V tensors and append at each step). Measure tokens/sec with and without the cache.

Then — before moving to Week 9 — run a **graph-break audit** on your full model. This is the bridge between Phase 1's expressive Python and Phase 2's compiled kernels:

```python
import torch._dynamo
torch._dynamo.config.verbose = True
compiled_model = torch.compile(model)
# run one forward pass and inspect the logs for "graph break" warnings
```

Any Python-level control flow, dynamic shapes, or non-tensor operations in your KV cache or RoPE implementation will cause graph breaks, making `torch.compile` in Week 10 generate slower code than eager mode. Fix the breaks now, at the architectural level, before you add FlashAttention on top of a leaky abstraction. Common culprits: Python `if` statements inside `forward()`, lists instead of tensors for the KV cache, and `.item()` calls that pull scalars to CPU.

**Discussion anchor:** Calculate the KV cache size for your model config at context length 2048 under MHA vs GQA-4 vs MQA. How does this translate to a maximum batch size on a 24 GB GPU? What graph breaks did you find, and how did you fix them?

---

#### Week 9 — Hardware-Aware Attention (FlashAttention 1 & 2)

**The question:** Why is standard attention memory-bound rather than compute-bound, and how does tiling fix it?

**Core references:**

- Dao et al. (2022), _FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness_: https://arxiv.org/abs/2205.14135
- Dao et al. (2023), _FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning_: https://arxiv.org/abs/2307.08691
- Blog: _Making Deep Learning Go Brrrr From First Principles_ (Horace He): https://horace.io/brrr_intro.html

**Deeper / optional:**

- Dao (2024), _FlashAttention-3_: https://arxiv.org/abs/2407.08608
- CUDA Mode lectures on memory hierarchies (YouTube): https://www.youtube.com/@cudamode

**Code milestone:** Implement a pure-PyTorch simulation of tiled online softmax. Walk through the IO complexity calculation: how many HBM reads/writes does standard attention do vs tiled attention? Switch your model to use `torch.nn.functional.scaled_dot_product_attention` (which calls FlashAttention under the hood on CUDA) and benchmark the speedup.

**Discussion anchor:** Derive the online softmax update rule. Why is it numerically identical to standard softmax? What is the tradeoff between tile size and SRAM usage?

---

#### Week 10 — Compiling Attention: FlexAttention & torch.compile

**The question:** Can we get FlashAttention-level performance without writing CUDA by hand?

**Core references:**

- PyTorch blog, _FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention_: https://pytorch.org/blog/flexattention/
- PyTorch docs, `torch.compile` overview: https://pytorch.org/docs/stable/torch.compiler.html
- Karpathy, _Let's reproduce GPT-2_ (video, covers compile and optimisation): https://youtu.be/l8pRSuU81PU

**Deeper / optional:**

- Cai et al. (2024), _FlashLight: Enabling Innovation in LLM Kernels via Compiler Infrastructure_: https://arxiv.org/abs/2412.07625
- Triton language documentation and tutorials: https://triton-lang.org/

**Code milestone:** Add `torch.compile` to your training loop. Use FlexAttention to implement a simple attention variant (e.g. sliding window / local attention mask). Profile with PyTorch profiler: compare compiled vs eager, and identify the remaining bottlenecks.

**Discussion anchor:** What does `torch.compile` actually do (trace, graph capture, kernel fusion)? What kinds of Python patterns break it? When would you reach for Triton instead?

---

#### Week 11 — Scaling, Numerical Precision & Training Dynamics

**The question:** How do you train stably in half precision, and what does the loss curve actually tell you?

**Core references:**

- Micikevicius et al. (2018), _Mixed Precision Training_: https://arxiv.org/abs/1710.03740
- Karpathy, _Let's reproduce GPT-2_ — the mixed precision and gradient scaling sections
- d2l.ai, Ch. 19 — Computational Performance: https://d2l.ai/chapter_computational-performance/

**Deeper / optional:**

- Dettmers et al. (2022), _LLM.int8()_ — quantisation context: https://arxiv.org/abs/2208.07339
- The BF16 vs FP16 explainer in the original LLaMA technical report appendix

**Code milestone:** Switch your training loop to BF16 mixed precision using `torch.autocast`. Add gradient clipping. Log loss curves and gradient norms. Try reproducing a small known result: train on TinyShakespeare to a target validation loss and compare your curve shape to Karpathy's reference.

**Discussion anchor:** Why does BF16 train more stably than FP16 out of the box? (Hint: exponent bits.) When would you prefer FP16? What is dynamic loss scaling and when do you need it?

---

#### Week 12 — Evaluation, Generation & the Complete System

**The question:** How do you know if your model is actually good?

**Core references:**

- Jurafsky & Martin, _Speech and Language Processing_, Ch. 3 — Language Modelling with N-grams (perplexity definition): https://web.stanford.edu/~jurafsky/slp3/
- Liang et al. (2022), _Holistic Evaluation of Language Models (HELM)_: https://arxiv.org/abs/2211.09110
- CS224n, LLM Evaluation lecture slides: http://web.stanford.edu/class/cs224n/

**Deeper / optional:**

- Srivastava et al. (2022), _Beyond the Imitation Game (BIG-Bench)_: https://arxiv.org/abs/2206.04615
- Biderman et al. (2023), _Pythia: A Suite for Analysing Large Language Models_: https://arxiv.org/abs/2304.01373

**Code milestone:** Full integration. Implement nucleus (top-p) sampling and temperature scaling for generation. Run `common/eval.py` on the final leaderboard snapshot. Each participant writes a short README in their implementation folder: architecture decisions, what they changed from the baseline, final loss.

**Discussion anchor:** Perplexity is easy to compute but what does it miss? When does a lower-perplexity model generate _worse_ text? What would you add to make this a stronger benchmark?

---

### 🎉 Final Demo Day — After Week 12

Each participant presents their complete system:

1. Architecture overview — what choices did you make and why?
2. Training curve — what did you learn from it?
3. A live generation demo
4. Final leaderboard position and post-mortem

Optionally: write up a shared group blog post or technical report summarising what you built. This is the CV artefact.

---

## Quick Reference: Resources by Type

### Videos (watch in roughly this order)

| Resource                         | Link                                                                 | When                     |
| -------------------------------- | -------------------------------------------------------------------- | ------------------------ |
| Karpathy — makemore series       | https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ | Pre-work / Weeks 1–2     |
| Karpathy — Let's build GPT       | https://youtu.be/kCc8FmEb1nY                                         | **Pre-work (mandatory)** |
| Karpathy — Let's reproduce GPT-2 | https://youtu.be/l8pRSuU81PU                                         | Weeks 10–11              |
| CUDA Mode lectures               | https://www.youtube.com/@cudamode                                    | Weeks 9–10 (optional)    |

### Textbooks (free online)

| Resource                | Link                                     | When        |
| ----------------------- | ---------------------------------------- | ----------- |
| d2l.ai                  | https://d2l.ai                           | Throughout  |
| Jurafsky & Martin, SLP3 | https://web.stanford.edu/~jurafsky/slp3/ | Weeks 1, 12 |

### Course Materials

| Resource        | Link                                  | When                       |
| --------------- | ------------------------------------- | -------------------------- |
| Stanford CS224n | http://web.stanford.edu/class/cs224n/ | Throughout (supplementary) |

### Essential Papers (in reading order)

| Paper                                                 | Link                             | Week |
| ----------------------------------------------------- | -------------------------------- | ---- |
| Vaswani et al. (2017) — Attention Is All You Need     | https://arxiv.org/abs/1706.03762 | 3–4  |
| Ba et al. (2016) — Layer Normalization                | https://arxiv.org/abs/1607.06450 | 4    |
| Shazeer (2020) — GLU Variants                         | https://arxiv.org/abs/2002.05202 | 5    |
| Su et al. (2021) — RoFormer / RoPE                    | https://arxiv.org/abs/2104.09864 | 7    |
| Ainslie et al. (2023) — GQA                           | https://arxiv.org/abs/2305.13245 | 8    |
| Dao et al. (2022) — FlashAttention                    | https://arxiv.org/abs/2205.14135 | 9    |
| Dao et al. (2023) — FlashAttention-2                  | https://arxiv.org/abs/2307.08691 | 9    |
| Micikevicius et al. (2018) — Mixed Precision Training | https://arxiv.org/abs/1710.03740 | 11   |

---

## Norms & Expectations

- **Come prepared.** Read the core references before the session. The presenter is not responsible for teaching basics the group was meant to pre-read.
- **Push your code before the session.** Even if it is broken. Especially if it is broken.
- **Ask why, not just how.** If you can run the code but cannot explain the gradient flow, that is an open question for the group.
- **Be kind about code.** Code review is about understanding, not judgement.
- **Missing a session is fine; missing two in a row means a check-in.** Momentum is the main risk for a group like this.

---

## Compute: Using the H100 Cluster

We have access to a cluster of H100 GPUs. The rule is simple: **use it thoughtfully, not wastefully**. The 12-week curriculum is designed so that most experimentation runs comfortably on a single consumer GPU or even CPU for the early weeks (TinyShakespeare fits in memory and trains in minutes). The H100s are for moments where scale actually teaches you something new.

### When to use the cluster

| Phase                       | Recommended Compute            | Why                                                                       |
| --------------------------- | ------------------------------ | ------------------------------------------------------------------------- |
| Weeks 1–5                   | Local CPU / single GPU         | Models are tiny; fast iteration matters more than scale                   |
| Weeks 6–8                   | Single H100 or cheap cloud GPU | First full training runs; KV cache profiling benefits from real VRAM      |
| Weeks 9–12                  | H100 (1–2 GPUs)                | FlashAttention benchmarks, mixed precision, meaningful throughput numbers |
| Post-curriculum experiments | H100 cluster (multi-GPU)       | DDP, tensor parallelism, larger models — see Phase 3 below                |

### Responsible cluster use

- **Always profile before scaling.** Run one step locally with `torch.profiler` before launching a cluster job. Most bugs are visible at small scale.
- **Use short-lived jobs.** Prefer many small experiments over one giant run. Set a `max_steps` budget before you launch.
- **Log everything.** Use Weights & Biases (free tier) or TensorBoard. A run you cannot reproduce is a run that did not happen.
- **Checkpoint frequently.** H100 jobs can be preempted. Save every N steps.
- **Share interesting runs with the group.** If you find something surprising on the cluster, bring it to the next session. That is the whole point.

### Suggested cluster experiments (within the 12-week curriculum)

- **Week 9:** Run a proper FlashAttention vs standard attention benchmark at sequence lengths 512 / 1024 / 4096 / 8192. Plot IO-bound vs compute-bound crossover.
- **Week 11:** Train the same model in FP32, BF16, and FP16. Compare loss curves, training time, and final perplexity. The H100's BF16 Tensor Core throughput makes this genuinely instructive.
- **Week 12:** Scale your model up one order of magnitude (e.g. from ~10M to ~100M parameters) and train on a larger dataset slice. See how your architecture decisions hold up.

### Cheap GPU alternatives

For personal tinkering outside of group sessions, a **Lambda Labs** or **Vast.ai** instance (RTX 3090 / 4090, ~$0.30–0.50/hr) is usually sufficient and avoids touching the shared cluster. Keep the H100s for things the group is doing together.

---

## What's Next — Keeping the Group Alive

The 12-week curriculum is a foundation, not a destination. The goal is for this group to become a **permanent fixture** — a place where, even when work is dominated by Power BI reports and data extractions that could have been a SQL query, there is something intellectually alive happening on the side.

After the first cohort completes, the group shifts format slightly: **one session every one or two weeks**, one paper or topic per session, rotating presenter. Lower overhead, higher autonomy. Below is a map of where you can go next, organised by theme.

---

### Track A — Going Deeper on the Transformer

The 12-week curriculum gives you the GPT architecture. Modern production LLMs diverge from that in interesting ways worth understanding from the inside.

**Mixture of Experts (MoE)**
Instead of every token passing through every FFN, MoE routes each token to a small subset of expert FFN blocks. This allows massive parameter counts at constant compute.

- Shazeer et al. (2017), _Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer_: https://arxiv.org/abs/1701.06538
- Jiang et al. (2024), _Mixtral of Experts_ (a clean modern implementation): https://arxiv.org/abs/2401.04088
- _Build it:_ Add a simple top-2 MoE FFN layer to your nanoGPT. Implement the routing and load-balancing auxiliary loss.

**State Space Models & Linear Attention**
Transformers have quadratic attention complexity. A wave of architectures — Mamba, RWKV, GLA — try to achieve transformer-quality at linear complexity. Understanding _why_ this is hard is as instructive as understanding the solutions.

- Gu & Dao (2023), _Mamba: Linear-Time Sequence Modelling with Selective State Spaces_: https://arxiv.org/abs/2312.00752
- Peng et al. (2023), _RWKV: Reinventing RNNs for the Transformer Era_: https://arxiv.org/abs/2305.13048
- Yang et al. (2024), _Gated Linear Attention_: https://arxiv.org/abs/2312.06635

**Speculative Decoding**
Autoregressive generation is slow because each token requires a full forward pass. Speculative decoding uses a small draft model to propose tokens that the large model verifies in parallel.

- Leviathan et al. (2023), _Fast Inference from Transformers via Speculative Decoding_: https://arxiv.org/abs/2211.17192
- _Build it:_ Implement a toy speculative decoder pairing your nanoGPT (verifier) with a 2-layer version of itself (drafter).

---

### Track B — Alignment, RLHF & Fine-Tuning

This is where the transformer stack meets human preferences — and where most production ML work on LLMs actually happens.

**Supervised Fine-Tuning (SFT)**
The simplest form of adaptation: continue training a pretrained model on a curated instruction-following dataset. A great first step that demonstrates how quickly a model's behaviour changes with targeted data.

- Wei et al. (2021), _Finetuned Language Models Are Zero-Shot Learners (FLAN)_: https://arxiv.org/abs/2109.01652
- Taori et al. (2023), _Alpaca: Instruction-following LLaMA_ (Stanford): https://crfm.stanford.edu/2023/03/13/alpaca.html
- _Build it:_ Fine-tune your Week 12 model on a small instruction dataset. Compare generation quality before and after.

**Reinforcement Learning from Human Feedback (RLHF)**
The technique behind ChatGPT-style alignment. A reward model is trained on human preference pairs, then used to fine-tune the LLM via PPO.

- Christiano et al. (2017), _Deep Reinforcement Learning from Human Preferences_: https://arxiv.org/abs/1706.03741
- Ziegler et al. (2019), _Fine-Tuning Language Models from Human Preferences_: https://arxiv.org/abs/1909.08593
- Ouyang et al. (2022), _Training language models to follow instructions with human feedback (InstructGPT)_: https://arxiv.org/abs/2203.02155
- _Understand it first:_ RLHF requires understanding PPO. Spend a session on Schulman et al. (2017), _Proximal Policy Optimization_: https://arxiv.org/abs/1707.06347

**Direct Preference Optimisation (DPO)**
A cleaner alternative to RLHF that bypasses the reward model entirely, treating preference alignment as a classification problem directly on the LLM.

- Rafailov et al. (2023), _Direct Preference Optimization: Your Language Model is Secretly a Reward Model_: https://arxiv.org/abs/2305.18290
- _This is one of the most elegant recent papers in alignment. The derivation is worth doing on a whiteboard._

**Parameter-Efficient Fine-Tuning (PEFT)**
Fine-tuning all parameters of a large model is expensive. PEFT methods freeze most of the model and adapt only a small number of new parameters.

- Hu et al. (2021), _LoRA: Low-Rank Adaptation of Large Language Models_: https://arxiv.org/abs/2106.09685
- Dettmers et al. (2023), _QLoRA: Efficient Fine-tuning of Quantized LLMs_: https://arxiv.org/abs/2305.14314
- _Build it:_ Implement LoRA from scratch on your attention projection matrices. Count the trainable parameters. Compare to full fine-tuning on the same task.

---

### Track C — Multimodality & Beyond Text

Once you understand the transformer deeply, extending it to other modalities is surprisingly tractable.

**Vision Transformers (ViT)**
Apply the transformer architecture directly to image patches. The architecture is almost identical to what you built — just with a different tokeniser.

- Dosovitskiy et al. (2020), _An Image is Worth 16x16 Words_: https://arxiv.org/abs/2010.11929
- _Build it:_ Swap your text tokeniser for a patch embedder and train on MNIST or CIFAR-10.

**Contrastive & Cross-Modal Learning (CLIP)**
Train a vision encoder and a text encoder jointly so that matching image-text pairs have similar representations. The foundation of most modern vision-language models.

- Radford et al. (2021), _Learning Transferable Visual Models From Natural Language Supervision (CLIP)_: https://arxiv.org/abs/2103.00020

**Diffusion Models**
Not a transformer topic per se, but deeply related — modern diffusion models (Stable Diffusion, FLUX) use transformer backbones (DiT). Understanding diffusion unlocks a huge part of the generative AI landscape.

- Ho et al. (2020), _Denoising Diffusion Probabilistic Models (DDPM)_: https://arxiv.org/abs/2006.11239
- Peebles & Xie (2022), _Scalable Diffusion Models with Transformers (DiT)_: https://arxiv.org/abs/2212.09748

---

### Track D — Systems & Scale

For those who want to go further on the hardware and distributed training side that the curriculum introduces in Weeks 9–11.

**Distributed Training**

- Rajbhandari et al. (2020), _ZeRO: Memory Optimizations Toward Training Trillion Parameter Models_: https://arxiv.org/abs/1910.02054
- Shoeybi et al. (2019), _Megatron-LM: Training Multi-Billion Parameter Language Models_: https://arxiv.org/abs/1909.08053
- _Build it:_ Take your nanoGPT and add PyTorch DDP (Distributed Data Parallel). Run a 2-GPU training job on the H100 cluster. Measure linear scaling.

**Quantisation**

- Dettmers et al. (2022), _LLM.int8()_ — 8-bit inference: https://arxiv.org/abs/2208.07339
- Frantar et al. (2022), _GPTQ: Accurate Post-Training Quantisation for GPT_: https://arxiv.org/abs/2210.17323

**Inference Optimisation**

- Kwon et al. (2023), _Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)_: https://arxiv.org/abs/2309.06180
- _This is how production LLM serving actually works. Read alongside the KV cache material from Week 8._

---

### How to Choose What's Next

After the first cohort, let the group vote on the next mini-arc at the final demo day. A suggested cadence:

- **Every 4–6 weeks:** one mini-arc of 3–4 sessions on a focused theme (e.g. "LoRA & DPO", or "Mamba vs Transformers", or "Distributed training on the cluster")
- **Interspersed:** single-session "paper of the week" discussions where one person presents a recent arXiv paper they found interesting — no code required
- **Annually:** revisit the 12-week curriculum with a new cohort, with existing members as mentors

---

## The Longer Game — From Reading Group to Research Group

Here is the honest ambition, written down so it does not stay just a vague feeling after the demo day drinks.

The 12-week curriculum makes you a highly competent reader and implementer. The tracks above make you broadly literate across the modern LLM landscape. But the real goal — the one worth stating plainly — is to build a group with a **research mentality**: people who do not just consume ideas but generate them, stress-test them, and occasionally produce something genuinely new.

That transition does not require anyone to become an academic. It requires three things: deep enough foundations to identify an interesting question, enough trust in the group to explore it together, and enough compute to actually run the experiment. We have all three.

### What "research mentality" means in practice

It means that when you read a paper and think _"but what if they had done X instead"_, you do not just move on — you bring it to the group. It means that when you notice something odd in a training curve or a surprising result in your ablation, you treat it as a data point rather than a nuisance. It means being comfortable saying "I don't know, let's find out" and then actually finding out.

Concretely, it looks like:

- **Replication first.** Before proposing a new idea, replicate the closest existing result at small scale. This is not busywork — it is how you discover whether the paper's claims hold outside the authors' setup.
- **Ablations as arguments.** Every architectural or training decision is a hypothesis. Run the experiment with and without the thing. The leaderboard culture from the 12-week curriculum is the seed of this habit.
- **Write things down.** A short internal write-up after each mini-arc — what we read, what we built, what we found surprising — is the primitive form of a research note. These accumulate into something.
- **Chase the interesting thread.** If three sessions in a row the discussion keeps circling back to the same open question, that is probably the question worth spending a month on.

### The upgrade path: reading group → research group

This is not a sharp transition but a gradual one. A rough map:

| Stage                         | Format                                      | Output                                                                |
| ----------------------------- | ------------------------------------------- | --------------------------------------------------------------------- |
| **Cohort 1 (now)**            | Weekly curriculum, 12 weeks                 | Working nanoGPT, leaderboard, CV project                              |
| **Post-cohort reading group** | Fortnightly, paper-per-session              | Shared reading notes, occasional code                                 |
| **Mini-research sprints**     | 4–6 week focused arcs with an open question | Experimental results, internal write-up                               |
| **Research group**            | Ongoing, self-directed, hypothesis-driven   | Potential preprints, collaborations, or just something genuinely cool |

The jump from mini-research sprints to a real research group happens when the group finds a question it cares about enough to spend real time on. That might be an efficiency trick for fine-tuning on limited compute, a curious phenomenon in how attention heads specialise, a new positional encoding variant, or something nobody has thought to look at yet because they were too busy reformatting pivot tables.

It does not need to be a Nature paper. It needs to be interesting, tractable, and something the group is actually excited to work on. The H100 cluster is there. The papers are there. The only thing that has ever stopped groups like this from doing real research is losing the habit of meeting.

### So: don't lose the habit

The reading group exists to protect a specific kind of time — time that belongs to curiosity rather than the sprint board. Even in the weeks where the day job is particularly absurd (and there will be weeks where it is very absurd), ninety minutes on a Wednesday where you are thinking about attention mechanisms and not about whether the KPI dashboard refreshes correctly is a genuine act of professional self-preservation.

In two years, the ambition is this: everyone in the group can read any major LLM paper, understand it deeply, implement the core idea over a weekend, and identify whether it opens a door worth walking through. That combination — rigour, speed, and taste — is rare. It is also, not coincidentally, exactly what makes someone dangerous in the best possible sense.

Keep meeting. Keep building. Keep asking why.

---

_Last updated: Week 0. See you on the other side of the pivot table._
