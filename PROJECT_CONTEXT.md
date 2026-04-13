# Model Quantization Project -- Full Context

## Goal

Build an impactful model quantization project with quantifiable outcomes to get hired at **Mondi**, a small company with a healthcare division that works on quantization.

The project needs to be resume-ready as a "current project" within **3-4 days**. The problem, solution, and outcome pointers must be solid and strong.

---

## Resume Bullet (target)

> Fine-tuned Gemma-2-2B on medical QA data using QLoRA (PEFT + BnB), then systematically evaluated 4 post-training quantization methods (GPTQ, AWQ, BnB-NF4, dynamic INT8) across 8/4/3-bit configs -- achieving **X% memory reduction** with only **Y% accuracy drop** on PubMedQA, benchmarked across latency, perplexity, and downstream task accuracy.

---

## Project Title

**"Quantization-Aware Compression of Gemma 2 for Medical QA: A Systematic Evaluation of PTQ Methods on Domain-Adapted LLMs"**

---

## User Background

- PyTorch + TensorFlow Mobile experience
- No custom CUDA kernel experience
- Has read ML research papers but never implemented one
- Decent linear algebra, not deep
- **No prior quantization experience** -- this is a learning project as well
- Prefers Gemma models, Google ecosystem
- Prefers being asked questions rather than having assumptions made

## Hardware & Budget

- **Laptop** (no dedicated GPU assumed)
- **Kaggle** 2x T4 16GB
- **Vertex AI credits**: ~$900 total, **$100-150 allocated** for this project
- Estimated spend: **$50-70** on Vertex AI (A100 40GB instances)
- Use Kaggle for all tasks (2xT4 handles Gemma 2 2B well), Vertex AI only if needed

---

## Model: Gemma 2 2B (Base)

- **Model ID**: `google/gemma-2-2b`
- **Total params**: ~2.6B
- **Architecture**: Standard transformer decoder with grouped-query attention
- **Context**: 8K tokens (8192)
- **Vocab**: 256,000 tokens
- **Dtype**: float16 (for T4 compatibility)
- **Prompt format (Gemma 2)**:
  ```
  <start_of_turn>user
  {question}<end_of_turn>
  <start_of_turn>model
  {answer}<end_of_turn>
  ```

### Why Gemma 2 2B

- Well-supported model with broad compatibility across all quantization methods
- Small enough to fine-tune and quantize on Kaggle 2xT4 without issues
- Standard PEFT works out of the box (no Unsloth or monkey-patches needed)
- Full quantization method compatibility (GPTQ, AWQ, BnB-NF4, BnB-INT8 all work)
- Fits easily on T4 16GB even in fp16

---

## The Plan

### Why This Specific Project

1. **Max learning, lower effort**: Touches every major quantization method without needing to invent one
2. **Quantifiable outcomes**: Produces tables of numbers -- perplexity, task accuracy, memory, latency
3. **Healthcare angle**: Evaluated on medical QA benchmarks (PubMedQA, MedQA) -- directly relevant to Mondi
4. **Novel question**: "Does QLoRA fine-tuning change which quantization method preserves quality best?" -- under-explored
5. **Practical**: Exactly what a healthcare company deploying LLMs needs to know

### Day 1: Learn + Setup + Data Prep

| # | Task | Details |
|---|---|---|
| 1 | Learn quantization fundamentals | Read: (a) HuggingFace quantization concepts page, (b) AWQ paper intro + method section, (c) GPTQ paper intro + method section. ~2-3 hrs. Understand *what* each does and *why*, not deep math. |
| 2 | Set up GitHub repo | Structure: `README.md`, `notebooks/`, `scripts/`, `results/`, `configs/`. Init git. |
| 3 | Set up Kaggle environment | Install: `transformers`, `peft`, `bitsandbytes`, `auto-gptq`, `autoawq`, `datasets`, `accelerate`. Verify GPU access. |
| 4 | Prepare medical datasets | Download PubMedQA, MedQA from HuggingFace datasets. Create train/eval splits. Format for fine-tuning using Gemma 2 prompt format (`<start_of_turn>`/`<end_of_turn>`). |
| 5 | Baseline: Run Gemma-2-2B FP16 | Run inference on medical benchmarks. Record: perplexity, accuracy, memory usage, latency. This is the control. |

### Day 2: QLoRA Fine-Tuning on Medical Data (via PEFT + BnB)

| # | Task | Details |
|---|---|---|
| 6 | QLoRA fine-tune Gemma-2-2B | Use standard PEFT `get_peft_model` with `load_in_4bit=True`. LoRA rank 16, target all attention + FFN layers. Train on medical QA data ~2 epochs. |
| 7 | Evaluate fine-tuned model | Run same benchmarks as baseline. Record all metrics. Expect improvement on medical tasks. |
| 8 | Save & merge LoRA weights | Save adapter, then merge into base model with `merge_and_unload()`. This is what gets quantized. |

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
| 15 | Analysis | Which method wins? Where's the quality cliff? Does medical fine-tuning change which quantization method is best? |
| 16 | Write README as mini-paper | Motivation, Method, Results (with tables/charts), Key Findings, How to Reproduce. |
| 17 | Push to GitHub | Clean code, add requirements.txt, push. |
| 18 | Optional: Upload quantized models to HuggingFace | Publish best-performing quantized medical Gemma 2 variants. |

---

## Quantization Methods Summary (for reference)

| Method | What It Does | Bit Widths | Tool |
|---|---|---|---|
| **GPTQ** | One-shot weight quantization using Hessian-based error compensation | 2-8 bit | `auto-gptq` / `transformers` |
| **AWQ** | Activation-aware weight quantization -- protects salient weight channels | 4 bit | `autoawq` |
| **BnB NF4** | NormalFloat4 quantization, designed for normal-distributed weights | 4 bit | `bitsandbytes` |
| **BnB INT8** | Mixed-precision decomposition, handles outlier features separately | 8 bit | `bitsandbytes` |

### Key Concepts to Understand

- **Post-Training Quantization (PTQ)**: Quantize after training. No retraining needed. GPTQ, AWQ are PTQ.
- **Quantization-Aware Training (QAT)**: Train with quantization in the loop. Better quality but expensive.
- **QLoRA**: Fine-tune with 4-bit quantized base + LoRA adapters. What we use for Day 2 (via PEFT + BnB).
- **Weight-only vs W+A quantization**: We're doing weight-only (easier, well-supported). Activation quantization is much harder.
- **Calibration data**: Small dataset used during quantization to determine optimal scales/zero-points. We'll use medical text as calibration data (another novel angle).
- **Perplexity**: Standard quality metric -- lower is better. Measures how "surprised" the model is by text.

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
| Gemma-2-2B FP16 (baseline) | 16 | | | | | | | |
| Gemma-2-2B-Med FP16 (fine-tuned) | 16 | | | | | | | |
| Gemma-2-2B-Med GPTQ-8bit | 8 | | | | | | | |
| Gemma-2-2B-Med GPTQ-4bit | 4 | | | | | | | |
| Gemma-2-2B-Med GPTQ-3bit | 3 | | | | | | | |
| Gemma-2-2B-Med AWQ-4bit | 4 | | | | | | | |
| Gemma-2-2B-Med BnB-NF4 | 4 | | | | | | | |
| Gemma-2-2B-Med BnB-INT8 | 8 | | | | | | | |

---

## Budget Estimate

| Resource | Hours | Cost |
|---|---|---|
| QLoRA fine-tuning (Kaggle 2xT4) | ~2-3 hrs | Free |
| Quantization runs (GPTQ/AWQ) | ~2-3 hrs | Free |
| Benchmarking all variants | ~3-4 hrs | Free |
| Buffer for debugging/reruns | ~4 hrs | Free |
| **Total** | | **Free (Kaggle)** |

---

## Key Libraries

```
transformers>=4.40.0
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
- All code should run on Kaggle (2x T4 16GB).
- Use **Gemma 2 2B base** (`google/gemma-2-2b`) as the model.
- Use **standard PEFT + BitsAndBytes** for QLoRA fine-tuning, NOT Unsloth.
- Use Gemma 2 prompt format: `<start_of_turn>user`/`<end_of_turn>`, NOT `<|turn>`/`<turn|>`.
- Model dtype is **float16** for T4 compatibility.
- When writing scripts, include proper logging of all metrics -- every number matters for the resume.
