# Quantization-Aware Compression of Gemma 2 for Medical QA

**A Systematic Evaluation of Post-Training Quantization Methods on Domain-Adapted LLMs**

---

## Research Question

> Does QLoRA fine-tuning on medical data change which quantization method best preserves model quality?

This is an under-explored question with direct practical implications: healthcare companies deploying quantized LLMs need to know whether domain adaptation changes the optimal compression strategy.

---

## Overview

We fine-tune **Gemma 2 2B** (`google/gemma-2-2b`) on combined medical QA data (PubMedQA + MedQA) using QLoRA (4-bit base + LoRA adapters) via **standard PEFT + BitsAndBytes**, merge the adapters back into the base model, then systematically apply 4 post-training quantization (PTQ) methods across multiple bit-width configurations. Every variant is benchmarked on the same evaluation suite to produce a direct comparison.

### Pipeline

```
Gemma 2 2B (FP16 baseline)
    |
    v
QLoRA Fine-Tune (PEFT + BnB, 4-bit base + LoRA adapters)
    |
    v
Merge LoRA into Base Model (merge_and_unload)
    |
    v
Quantize with 4 PTQ Methods (GPTQ 8/4/3-bit, AWQ 4-bit, BnB-NF4, BnB-INT8)
    |
    v
Benchmark All Variants (perplexity, accuracy, VRAM, throughput)
    |
    v
Analysis & Comparison
```

### Why This Project

1. **Quantifiable outcomes** -- produces concrete tables of numbers: perplexity, task accuracy, memory savings, latency
2. **Healthcare angle** -- evaluated on medical QA benchmarks (PubMedQA, MedQA), directly relevant to clinical NLP deployment
3. **Novel question** -- "Does domain-specific fine-tuning change which quantization method preserves quality best?" is under-explored in literature
4. **Practical** -- exactly the kind of analysis a healthcare company deploying LLMs needs before choosing a compression strategy
5. **Medical calibration data** -- GPTQ and AWQ use PubMedQA medical text (not generic C4/WikiText) as calibration data, which is another novel angle

---

## Model

**Gemma 2 2B** (`google/gemma-2-2b`) -- Google's efficient dense model (base, NOT instruction-tuned):

| Property | Value |
|----------|-------|
| Model ID | `google/gemma-2-2b` |
| Parameters | ~2.6B |
| Architecture | Standard transformer decoder with grouped-query attention |
| Context Length | 8,192 tokens |
| Vocabulary | 256,000 tokens |
| Dtype | float16 (for T4 GPU compatibility) |

### Prompt Format (Gemma 2)

```
<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
{answer}<end_of_turn>
```

All training data formatting, evaluation prompts, and inference code use this exact format consistently across every file.

---

## Quantization Methods

| Method | What It Does | Bit Widths | Tool |
|--------|-------------|-----------|------|
| **GPTQ** | One-shot weight quantization using Hessian-based error compensation. Calibrates on a small dataset to minimize quantization error layer-by-layer. | 8, 4, 3 | `auto-gptq` / `transformers` `GPTQConfig` |
| **AWQ** | Activation-aware weight quantization -- identifies and protects salient weight channels that matter most for activations. | 4 | `autoawq` |
| **BnB NF4** | NormalFloat4 quantization, designed for normally-distributed weights. Uses double quantization for extra compression. Runtime-only (no saved model). | 4 | `bitsandbytes` |
| **BnB INT8** | Mixed-precision decomposition that handles outlier features separately in FP16 while quantizing the rest to INT8. Runtime-only. | 8 | `bitsandbytes` |

### Key Concepts

- **Post-Training Quantization (PTQ)**: Quantize after training -- no retraining needed. All 4 methods above are PTQ.
- **QLoRA**: Fine-tune with a 4-bit quantized base model + trainable LoRA adapters. Used in the fine-tuning step (Day 2), not the final quantization.
- **Weight-only quantization**: We quantize weights only (not activations). This is simpler, well-supported, and standard for LLM deployment.
- **Calibration data**: A small dataset used during GPTQ/AWQ quantization to determine optimal scales and zero-points. We intentionally use **medical text** from PubMedQA as calibration data to see if domain-matched calibration helps.
- **Perplexity**: Standard language model quality metric -- lower is better. Measures how "surprised" the model is by text.

---

## Datasets

### Training Data

| Dataset | Source | Split Used | Examples | Purpose |
|---------|--------|-----------|----------|---------|
| **PubMedQA** | `pubmed_qa` (config: `pqa_labeled`) | `train` | 1,000 | Medical yes/no/maybe QA with research paper contexts |
| **MedQA** | `GBaker/MedQA-USMLE-4-options` | `train` | ~10,000 | USMLE-style multiple choice medical questions |

Combined training set: ~11,000 examples. Validation: MedQA `test` split (PubMedQA has no separate validation split).

### Evaluation Benchmarks

| Metric | Dataset/Method | What It Measures |
|--------|---------------|-----------------|
| Perplexity (WikiText-2) | `wikitext-2-raw-v1` test split | General language modeling quality |
| Perplexity (Medical) | PubMedQA contexts | Domain-specific quality retention after quantization |
| PubMedQA Accuracy | `pubmed_qa` `pqa_labeled` (200 samples) | Medical yes/no/maybe question answering accuracy |
| Tokens/sec | Custom generation benchmark | Inference throughput |
| Peak VRAM (GB) | `torch.cuda.max_memory_allocated` | GPU memory footprint |

---

## Fine-Tuning Configuration (QLoRA)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base quantization | 4-bit NF4 | QLoRA standard -- keeps base frozen in 4-bit |
| LoRA rank (r) | 16 | Good balance of capacity vs. parameter count |
| LoRA alpha | 32 | 2x rank (standard scaling) |
| LoRA dropout | 0.05 | Light regularization |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All 7 attention + FFN projection layers |
| Epochs | 2 | Sufficient for adapter convergence on this data size |
| Batch size | 2 (per device) | T4 memory constraint |
| Gradient accumulation | 8 steps | Effective batch size = 16 |
| Learning rate | 2e-4 | Standard for QLoRA |
| Scheduler | Cosine with 50 warmup steps | Smooth LR decay |
| Optimizer | AdamW 8-bit | Memory-efficient optimizer |
| Max sequence length | 512 | Balances context coverage vs. memory |
| Precision | FP16 | T4 does not support BF16 well |
| Gradient checkpointing | Enabled | Trades compute for memory |

After training, LoRA adapters are **merged into the base model** via `model.merge_and_unload()` to produce a full FP16 checkpoint. This merged model is the input to every quantization method.

---

## Results

| Model Variant | Bits | Perplexity (Wiki) | Perplexity (Med) | PubMedQA Acc | VRAM (GB) | Tok/s |
|---|---|---|---|---|---|---|
| Gemma-2-2B FP16 (baseline) | 16 | -- | -- | -- | -- | -- |
| Gemma-2-2B-Med FP16 (fine-tuned) | 16 | -- | -- | -- | -- | -- |
| Gemma-2-2B-Med GPTQ-8bit | 8 | -- | -- | -- | -- | -- |
| Gemma-2-2B-Med GPTQ-4bit | 4 | -- | -- | -- | -- | -- |
| Gemma-2-2B-Med GPTQ-3bit | 3 | -- | -- | -- | -- | -- |
| Gemma-2-2B-Med AWQ-4bit | 4 | -- | -- | -- | -- | -- |
| Gemma-2-2B-Med BnB-NF4 | 4 | -- | -- | -- | -- | -- |
| Gemma-2-2B-Med BnB-INT8 | 8 | -- | -- | -- | -- | -- |

*Results will be filled as experiments complete.*

---

## Project Structure

```
.
├── README.md                              # This file -- complete project documentation
├── CLAUDE.md                              # AI assistant instructions (codebase conventions)
├── requirements.txt                       # Python dependencies
│
├── notebooks/                             # Ordered, self-contained Jupyter notebooks
│   ├── 01_data_preparation.ipynb          # Download & format datasets + FP16 baseline eval
│   ├── 02_qlora_finetune.ipynb            # QLoRA fine-tuning + merge + fine-tuned eval
│   ├── 03_quantize_and_benchmark.ipynb    # Apply all 4 PTQ methods + benchmark each
│   └── 04_analysis.ipynb                  # Results tables, charts, key findings
│
├── scripts/                               # Reusable Python modules (called by notebooks)
│   ├── data_prep.py                       # Dataset loading, formatting, calibration data
│   ├── evaluate.py                        # Perplexity, PubMedQA accuracy, speed, memory
│   └── quantize.py                        # GPTQ, AWQ, BnB-NF4, BnB-INT8 runners
│
├── configs/                               # Declarative hyperparameters
│   ├── qlora_config.yaml                  # QLoRA fine-tuning config
│   └── quantization_configs.yaml          # PTQ method configs (bits, group_size, etc.)
│
├── results/                               # Generated at runtime (gitignored)
│   ├── tables/all_results.csv             # Canonical results CSV (one row per model variant)
│   └── figures/                           # Charts from notebook 04
│
└── data/                                  # Dataset cache (gitignored)
```

### How the Code Fits Together

The codebase is small and pipeline-shaped. The 4 ordered notebooks drive the end-to-end flow and call into reusable functions in `scripts/`:

- **`scripts/data_prep.py`** -- Loads PubMedQA and MedQA from HuggingFace, formats them with the Gemma 2 prompt template (`<start_of_turn>`/`<end_of_turn>`). Also supplies `prepare_calibration_data()` which feeds medical text into GPTQ/AWQ calibration.

- **`scripts/quantize.py`** -- One function per PTQ method (`quantize_gptq()`, `quantize_awq()`, `load_bnb_nf4()`, `load_bnb_int8()`). GPTQ and AWQ produce saved quantized models; BnB variants are runtime-only. `quantize_all()` runs every method sequentially with memory cleanup between runs.

- **`scripts/evaluate.py`** -- Metric functions plus `run_full_evaluation()` which appends one row per model variant to `results/tables/all_results.csv` (deduplicates by model name on re-runs). This CSV is the canonical results store that feeds notebook 04 and the README table.

- **`configs/`** -- Declarative YAML configs for fine-tuning and quantization hyperparameters. These mirror what the notebooks use and should be kept in sync.

---

## Execution Plan

### Notebook 01: Data Preparation + Baseline

1. Install dependencies, login to HuggingFace
2. Download PubMedQA (`pqa_labeled`) and MedQA (`GBaker/MedQA-USMLE-4-options`)
3. Format both datasets with Gemma 2 prompt templates
4. Load base Gemma 2 2B in 4-bit and run baseline evaluation:
   - WikiText-2 perplexity (256 samples)
   - Medical text perplexity (256 PubMedQA contexts)
   - PubMedQA yes/no/maybe accuracy (200 samples)
   - Inference speed (5 runs x 50 tokens)
5. Save baseline results to `results/tables/all_results.csv`

### Notebook 02: QLoRA Fine-Tuning

1. Load Gemma 2 2B in 4-bit with BitsAndBytes NF4
2. Add LoRA adapters targeting all 7 projection layers
3. Load and format combined training data (PubMedQA + MedQA)
4. Train for 2 epochs with SFTTrainer (effective batch size 16)
5. Save LoRA adapter checkpoint
6. Merge LoRA into base model (`merge_and_unload()`) and save merged checkpoint
7. Sanity check: load merged model and test generation
8. Run full evaluation on fine-tuned model and save results

### Notebook 03: Quantize & Benchmark

1. Prepare medical calibration data (64 PubMedQA contexts)
2. For each quantization method:
   - Load/quantize the merged fine-tuned model
   - Run full evaluation suite (perplexity, accuracy, speed, memory)
   - Save results to CSV
   - Free memory before next method
3. Methods run in order: GPTQ 8-bit, GPTQ 4-bit, GPTQ 3-bit, AWQ 4-bit, BnB NF4, BnB INT8

### Notebook 04: Analysis

1. Load all results from CSV
2. Generate formatted comparison table (with % change vs. FP16 baseline)
3. Plot accuracy vs. memory tradeoff scatter chart
4. Plot perplexity comparison bar charts (WikiText-2 and medical)
5. Plot inference speed comparison
6. Auto-generate key findings summary

---

## Hardware

| Resource | Usage | Cost |
|----------|-------|------|
| Kaggle T4 16GB | Fine-tuning, quantization, benchmarking | Free |
| Any CPU | Notebook 04 analysis | Free |

Gemma 2 2B fits comfortably on a single T4 16GB in FP16 (~5 GB), and all quantized variants use even less memory. No A100 or multi-GPU setup required.

---

## Setup & Reproduction

### Prerequisites

```bash
pip install -r requirements.txt
huggingface-cli login   # Needs access to google/gemma-2-2b
```

### Key Dependencies

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
sentencepiece
protobuf
pandas
tqdm
matplotlib
```

### Running

Run the 4 notebooks in order on Kaggle (T4 16GB). Each notebook is self-contained and installs its own dependencies at the top.

```
01_data_preparation.ipynb  ->  02_qlora_finetune.ipynb  ->  03_quantize_and_benchmark.ipynb  ->  04_analysis.ipynb
```

The scripts in `scripts/` are library modules -- they're imported by the notebooks, not run as standalone scripts (except `data_prep.py` which can be run with `python scripts/data_prep.py` to pre-download datasets).

---

## Key Findings

*To be completed after experiments.*

---

## License

MIT
