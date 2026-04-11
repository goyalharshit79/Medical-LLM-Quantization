# Model Quantization Project — Full Context

## Goal

Build an impactful model quantization project with quantifiable outcomes to get hired at **Mondi**, a small company with a healthcare division that works on quantization.

The project needs to be resume-ready as a "current project" within **3-4 days**. The problem, solution, and outcome pointers must be solid and strong.

---

## Resume Bullet (target)

> Fine-tuned Gemma-4-E4B on medical QA data using QLoRA (Unsloth), then systematically evaluated 4 post-training quantization methods (GPTQ, AWQ, BnB-NF4, dynamic INT8) across 8/4/3-bit configs — achieving **X% memory reduction** with only **Y% accuracy drop** on PubMedQA, benchmarked across latency, perplexity, and downstream task accuracy.

---

## Project Title

**"Quantization-Aware Compression of Gemma 4 for Medical QA: A Systematic Evaluation of PTQ Methods on Domain-Adapted LLMs"**

---

## User Background

- PyTorch + TensorFlow Mobile experience
- No custom CUDA kernel experience
- Has read ML research papers but never implemented one
- Decent linear algebra, not deep
- **No prior quantization experience** — this is a learning project as well
- Prefers Gemma models, Google ecosystem
- Prefers being asked questions rather than having assumptions made

## Hardware & Budget

- **Laptop** (no dedicated GPU assumed)
- **Kaggle / Google Colab** free tier (T4 16GB typically)
- **Vertex AI credits**: ~$900 total, **$100-150 allocated** for this project
- Estimated spend: **$50-70** on Vertex AI (A100 40GB instances)
- Use Colab free tier for lightweight tasks (data prep, small evals), Vertex AI for GPU-heavy work

---

## Model: Gemma 4 E4B (Base)

- **Model ID**: `google/gemma-4-E4B`
- **Total params**: ~8B (7,996,156,490) — **4.5B effective** (rest is PLE embedding tables)
- **Layers**: 42 decoder layers
- **Hidden size**: 2560
- **Attention**: 8 query heads, 2 KV heads (GQA 4:1 ratio). Head dim 256 (sliding) / 512 (global)
- **Architecture**: Hybrid sliding window (512 tokens) + full global attention. Pattern: 5 sliding → 1 global (7 global layers total)
- **Per-Layer Embeddings (PLE)**: Each decoder layer has its own token embedding table (256-dim). Large in storage, cheap at inference (just lookups)
- **KV sharing**: 18 layers share key-value projections
- **Context**: 128K tokens (131,072)
- **Vocab**: 262,144 tokens
- **Dtype**: bfloat16
- **Multimodal**: Supports text, image, audio — we only use text
- **Prompt format (Gemma 4)**:
  ```
  <|turn>user
  {question}<turn|>
  <|turn>model
  {answer}<turn|>
  ```

### Why E4B over Gemma 2B

- Latest architecture (April 2026) — shows you're current
- Slightly larger (4.5B effective vs 2B) — more meaningful compression results
- Novel architecture features (PLE, hybrid attention, KV sharing) — interesting quantization behavior
- Full quantization method compatibility (GPTQ, AWQ, BnB-NF4, BnB-INT8 all work)
- Fits on T4 16GB with QLoRA, A100 40GB for heavier tasks

---

## The Plan

### Why This Specific Project

1. **Max learning, lower effort**: Touches every major quantization method without needing to invent one
2. **Quantifiable outcomes**: Produces tables of numbers — perplexity, task accuracy, memory, latency
3. **Healthcare angle**: Evaluated on medical QA benchmarks (PubMedQA, MedQA) — directly relevant to Mondi
4. **Novel question**: "Does QLoRA fine-tuning change which quantization method preserves quality best?" — under-explored
5. **Practical**: Exactly what a healthcare company deploying LLMs needs to know
6. **Cutting-edge model**: Gemma 4 E4B released April 2, 2026 — one of the first quantization studies on this architecture

### Day 1: Learn + Setup + Data Prep

| # | Task | Details |
|---|---|---|
| 1 | Learn quantization fundamentals | Read: (a) HuggingFace quantization concepts page, (b) AWQ paper intro + method section, (c) GPTQ paper intro + method section. ~2-3 hrs. Understand *what* each does and *why*, not deep math. |
| 2 | Set up GitHub repo | Structure: `README.md`, `notebooks/`, `scripts/`, `results/`, `configs/`. Init git. |
| 3 | Set up Vertex AI / Colab environment | Install: `unsloth`, `transformers>=5.5.0`, `bitsandbytes`, `auto-gptq`, `autoawq`, `datasets`, `accelerate`. Verify GPU access. |
| 4 | Prepare medical datasets | Download PubMedQA, MedQA from HuggingFace datasets. Create train/eval splits. Format for fine-tuning using Gemma 4 prompt format (`<|turn>user`/`<turn|>`). |
| 5 | Baseline: Run Gemma-4-E4B BF16 | Run inference on medical benchmarks. Record: perplexity, accuracy, memory usage, latency. This is the control. |

### Day 2: QLoRA Fine-Tuning on Medical Data (via Unsloth)

| # | Task | Details |
|---|---|---|
| 6 | QLoRA fine-tune Gemma-4-E4B | Use Unsloth `FastLanguageModel` with `load_in_4bit=True`. LoRA rank 16, target all attention + FFN layers. Train on medical QA data ~2 epochs. Use Vertex AI if Colab times out. |
| 7 | Evaluate fine-tuned model | Run same benchmarks as baseline. Record all metrics. Expect improvement on medical tasks. |
| 8 | Save & merge LoRA weights | Save adapter, then merge into base model → full BF16 fine-tuned model. This is what gets quantized. |

### Day 3: Quantize + Benchmark

| # | Task | Details |
|---|---|---|
| 9 | Quantize with GPTQ | 8-bit, 4-bit, 3-bit variants. Use medical calibration data. |
| 10 | Quantize with AWQ | 4-bit. Use AutoAWQ. |
| 11 | Quantize with BnB NF4 | Load with `load_in_4bit=True`, NormalFloat4. |
| 12 | Quantize with dynamic INT8 | `load_in_8bit=True` via bitsandbytes. |
| 13 | Benchmark ALL variants | For each quantized model, record: perplexity (WikiText-2 + medical text), PubMedQA accuracy, MedQA accuracy, model size (GB), peak VRAM, tokens/sec. |

### Day 4: Analysis + Write-up + Publish

| # | Task | Details |
|---|---|---|
| 14 | Build results table | Compile all numbers into a comparison table. |
| 15 | Analysis | Which method wins? Where's the quality cliff? Does medical fine-tuning change which quantization method is best? How does PLE affect quantization? (Novel angles.) |
| 16 | Write README as mini-paper | Motivation, Method, Results (with tables/charts), Key Findings, How to Reproduce. |
| 17 | Push to GitHub | Clean code, add requirements.txt, push. |
| 18 | Optional: Upload quantized models to HuggingFace | Publish best-performing quantized medical Gemma 4 variants. |

---

## Quantization Methods Summary (for reference)

| Method | What It Does | Bit Widths | Tool |
|---|---|---|---|
| **GPTQ** | One-shot weight quantization using Hessian-based error compensation | 2-8 bit | `auto-gptq` / `transformers` |
| **AWQ** | Activation-aware weight quantization — protects salient weight channels | 4 bit | `autoawq` |
| **BnB NF4** | NormalFloat4 quantization, designed for normal-distributed weights | 4 bit | `bitsandbytes` |
| **BnB INT8** | Mixed-precision decomposition, handles outlier features separately | 8 bit | `bitsandbytes` |

### Key Concepts to Understand

- **Post-Training Quantization (PTQ)**: Quantize after training. No retraining needed. GPTQ, AWQ are PTQ.
- **Quantization-Aware Training (QAT)**: Train with quantization in the loop. Better quality but expensive.
- **QLoRA**: Fine-tune with 4-bit quantized base + LoRA adapters. What we use for Day 2 (via Unsloth).
- **Weight-only vs W+A quantization**: We're doing weight-only (easier, well-supported). Activation quantization is much harder.
- **Calibration data**: Small dataset used during quantization to determine optimal scales/zero-points. We'll use medical text as calibration data (another novel angle).
- **Perplexity**: Standard quality metric — lower is better. Measures how "surprised" the model is by text.
- **Per-Layer Embeddings (PLE)**: Gemma 4's architecture feature — each layer has its own embedding table. These are large on disk but just lookups at inference. Interesting question: how do quantization methods handle these?

---

## Evaluation Metrics

| Metric | What It Measures | Tools |
|---|---|---|
| Perplexity (WikiText-2) | General language modeling quality | Custom eval script |
| Perplexity (medical text) | Domain-specific quality retention | Custom eval script |
| PubMedQA accuracy | Medical question answering | Custom prompt-based eval |
| MedQA accuracy | Medical multiple choice | Custom prompt-based eval |
| Model size (GB) | Disk/memory footprint | `os.path.getsize` / HF |
| Peak VRAM (GB) | GPU memory during inference | `torch.cuda.max_memory_allocated` |
| Tokens/sec | Inference throughput | Custom timing |

---

## Expected Results Table (to be filled)

| Model Variant | Bits | Size (GB) | Perplexity (Wiki) | Perplexity (Med) | PubMedQA Acc | MedQA Acc | VRAM (GB) | Tok/s |
|---|---|---|---|---|---|---|---|---|
| Gemma-4-E4B BF16 (baseline) | 16 | | | | | | | |
| Gemma-4-E4B-Med BF16 (fine-tuned) | 16 | | | | | | | |
| Gemma-4-E4B-Med GPTQ-8bit | 8 | | | | | | | |
| Gemma-4-E4B-Med GPTQ-4bit | 4 | | | | | | | |
| Gemma-4-E4B-Med GPTQ-3bit | 3 | | | | | | | |
| Gemma-4-E4B-Med AWQ-4bit | 4 | | | | | | | |
| Gemma-4-E4B-Med BnB-NF4 | 4 | | | | | | | |
| Gemma-4-E4B-Med BnB-INT8 | 8 | | | | | | | |

---

## Budget Estimate

| Resource | Hours | Cost |
|---|---|---|
| QLoRA fine-tuning (A100 40GB) | ~3-4 hrs | ~$12-16 |
| Quantization runs (GPTQ/AWQ) | ~3-4 hrs | ~$12-16 |
| Benchmarking all variants | ~4-6 hrs | ~$16-24 |
| Buffer for debugging/reruns | ~4 hrs | ~$16 |
| **Total** | | **~$55-72** |

---

## Key Libraries

```
unsloth
transformers>=5.5.0
peft>=0.9.0
bitsandbytes>=0.43.0
auto-gptq>=0.7.0
autoawq>=0.2.0
datasets
accelerate
torch>=2.1.0
trl
```

---

## Instructions for the AI assistant in the next session

- The user has no prior quantization experience. Explain concepts as they come up, briefly.
- The user prefers being asked questions rather than having assumptions made.
- Help the user execute the day-by-day plan above. Start from wherever they say they are.
- Prioritize practical, working code over perfection. Get experiments running fast.
- All code should run on Colab/Kaggle (T4 16GB) or Vertex AI (A100 40GB).
- Use **Gemma 4 E4B base** (`google/gemma-4-E4B`) as the model. NOT Gemma 2B.
- Use **Unsloth** for QLoRA fine-tuning, NOT vanilla PEFT.
- Use Gemma 4 prompt format: `<|turn>user`/`<turn|>`, NOT `<start_of_turn>`/`<end_of_turn>`.
- Model dtype is **bfloat16**, not float16.
- When writing scripts, include proper logging of all metrics — every number matters for the resume.
