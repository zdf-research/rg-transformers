# Future Research Tracks

The 13-week curriculum is a foundation, not a destination. The goal is for this group to become a **permanent fixture** — intellectually alive even when work is dominated by Power BI reports and data extractions.

After the first cohort, the format shifts: one session every one or two weeks, one paper or topic per session, rotating presenter. Below is a map of where you can go next.

---

## Track A — Going Deeper on the Transformer

Modern production LLMs diverge from the GPT architecture in interesting ways.

### Mixture of Experts (MoE)
Instead of every token passing through every FFN, MoE routes each token to a small subset of expert FFN blocks.

- Shazeer et al. (2017), _Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer_: https://arxiv.org/abs/1701.06538
- Jiang et al. (2024), _Mixtral of Experts_: https://arxiv.org/abs/2401.04088
- _Build it:_ Add a simple top-2 MoE FFN layer to the group project. Implement routing and load-balancing auxiliary loss.

### State Space Models & Linear Attention
Transformers have quadratic attention complexity. Architectures like Mamba, RWKV, and GLA try to achieve transformer quality at linear complexity.

- Gu & Dao (2023), _Mamba: Linear-Time Sequence Modelling with Selective State Spaces_: https://arxiv.org/abs/2312.00752
- Peng et al. (2023), _RWKV: Reinventing RNNs for the Transformer Era_: https://arxiv.org/abs/2305.13048
- Yang et al. (2024), _Gated Linear Attention_: https://arxiv.org/abs/2312.06635

### Speculative Decoding
Autoregressive generation is slow because each token requires a full forward pass.

- Leviathan et al. (2023), _Fast Inference from Transformers via Speculative Decoding_: https://arxiv.org/abs/2211.17192
- _Build it:_ Implement a toy speculative decoder pairing the group model (verifier) with a 2-layer version of itself (drafter).

---

## Track B — Alignment, RLHF & Fine-Tuning

This is where the transformer meets human preferences — where most production ML work on LLMs actually happens.

### Supervised Fine-Tuning (SFT)
- Wei et al. (2021), _Finetuned Language Models Are Zero-Shot Learners (FLAN)_: https://arxiv.org/abs/2109.01652
- Taori et al. (2023), _Alpaca_: https://crfm.stanford.edu/2023/03/13/alpaca.html
- _Build it:_ Fine-tune the group project model on a small instruction dataset. Compare generation quality before and after.

### Reinforcement Learning from Human Feedback (RLHF)
- Christiano et al. (2017), _Deep RL from Human Preferences_: https://arxiv.org/abs/1706.03741
- Ziegler et al. (2019), _Fine-Tuning Language Models from Human Preferences_: https://arxiv.org/abs/1909.08593
- Ouyang et al. (2022), _InstructGPT_: https://arxiv.org/abs/2203.02155
- _Prerequisite:_ Schulman et al. (2017), _Proximal Policy Optimization_: https://arxiv.org/abs/1707.06347

### Direct Preference Optimisation (DPO)
- Rafailov et al. (2023), _DPO: Your Language Model is Secretly a Reward Model_: https://arxiv.org/abs/2305.18290
- _This is one of the most elegant recent papers in alignment. The derivation is worth doing on a whiteboard._

### Parameter-Efficient Fine-Tuning (PEFT)
- Hu et al. (2021), _LoRA: Low-Rank Adaptation of Large Language Models_: https://arxiv.org/abs/2106.09685
- Dettmers et al. (2023), _QLoRA_: https://arxiv.org/abs/2305.14314
- _Build it:_ Implement LoRA on attention projection matrices. Count trainable parameters. Compare to full fine-tuning.

---

## Track C — Multimodality & Beyond Text

Once you understand the transformer deeply, extending it to other modalities is surprisingly tractable.

### Vision Transformers (ViT)
- Dosovitskiy et al. (2020), _An Image is Worth 16x16 Words_: https://arxiv.org/abs/2010.11929
- _Build it:_ Swap the text tokenizer for a patch embedder and train on MNIST or CIFAR-10.

### Contrastive & Cross-Modal Learning (CLIP)
- Radford et al. (2021), _CLIP_: https://arxiv.org/abs/2103.00020

### Diffusion Models
- Ho et al. (2020), _DDPM_: https://arxiv.org/abs/2006.11239
- Peebles & Xie (2022), _DiT_: https://arxiv.org/abs/2212.09748

---

## Track D — Systems & Scale

For those who want to go further on hardware and distributed training.

### Distributed Training
- Rajbhandari et al. (2020), _ZeRO_: https://arxiv.org/abs/1910.02054
- Shoeybi et al. (2019), _Megatron-LM_: https://arxiv.org/abs/1909.08053
- _Build it:_ Add PyTorch DDP to the group model. Run a 2-GPU training job on the H100 cluster. Measure linear scaling.

### Quantisation
- Dettmers et al. (2022), _LLM.int8()_: https://arxiv.org/abs/2208.07339
- Frantar et al. (2022), _GPTQ_: https://arxiv.org/abs/2210.17323

### Inference Optimisation
- Kwon et al. (2023), _vLLM / PagedAttention_: https://arxiv.org/abs/2309.06180
- _Read alongside the KV cache material from Week 8._

---

## How to Choose What's Next

After the first cohort, let the group vote on the next mini-arc at the final demo day.

- **Every 4–6 weeks:** one mini-arc of 3–4 sessions on a focused theme (e.g. "LoRA & DPO", "Mamba vs Transformers", "Distributed training on the cluster")
- **Interspersed:** single-session "paper of the week" — no code required
- **Annually:** revisit the 13-week curriculum with a new cohort, with existing members as mentors
