# Quantization-Aware Compression of Gemma 2 for Medical QA

A systematic evaluation of post-training quantization (PTQ) methods on a domain-adapted LLM for medical question answering.

## Research Question

> Does QLoRA fine-tuning on medical data change which quantization method best preserves model quality?

## Overview

We fine-tune **Gemma 2 2B** (`google/gemma-2-2b`) on medical QA data using QLoRA via **standard PEFT + BitsAndBytes**, then systematically evaluate 4 post-training quantization methods across multiple bit-width configurations:

| Method | Bit Widths | Tool |
|--------|-----------|------|
| **GPTQ** | 8, 4, 3 | `auto-gptq` / `transformers` |
| **AWQ** | 4 | `autoawq` |
| **BnB NF4** | 4 | `bitsandbytes` |
| **BnB INT8** | 8 | `bitsandbytes` |

All variants are benchmarked on **PubMedQA** and **MedQA** for downstream accuracy, plus perplexity, memory usage, and inference latency.

## Model

**Gemma 2 2B** (`google/gemma-2-2b`) — Google's efficient dense model:
- ~2.6B parameters
- Standard transformer decoder with grouped-query attention
- 8K token context window, 256K vocabulary
- float16 dtype (for T4 compatibility)

## Results

| Model Variant | Bits | Perplexity (Wiki) | PubMedQA Acc | VRAM (GB) | Tok/s |
|---|---|---|---|---|---|
| Gemma-2-2B FP16 (baseline) | 16 | -- | -- | -- | -- |
| Gemma-2-2B-Med FP16 (fine-tuned) | 16 | -- | -- | -- | -- |
| Gemma-2-2B-Med GPTQ-8bit | 8 | -- | -- | -- | -- |
| Gemma-2-2B-Med GPTQ-4bit | 4 | -- | -- | -- | -- |
| Gemma-2-2B-Med GPTQ-3bit | 3 | -- | -- | -- | -- |
| Gemma-2-2B-Med AWQ-4bit | 4 | -- | -- | -- | -- |
| Gemma-2-2B-Med BnB-NF4 | 4 | -- | -- | -- | -- |
| Gemma-2-2B-Med BnB-INT8 | 8 | -- | -- | -- | -- |

*Results will be filled as experiments complete.*

## Project Structure

```
.
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Download & format medical QA datasets + baseline eval
│   ├── 02_qlora_finetune.ipynb        # QLoRA fine-tuning via PEFT + BitsAndBytes
│   ├── 03_quantize_and_benchmark.ipynb # Quantization + full benchmark suite
│   └── 04_analysis.ipynb              # Results analysis & visualization
├── scripts/
│   ├── data_prep.py                   # Dataset loading & formatting utilities
│   ├── evaluate.py                    # Perplexity & accuracy evaluation
│   └── quantize.py                    # Quantization runners for each method
├── configs/
│   ├── qlora_config.yaml              # QLoRA hyperparameters
│   └── quantization_configs.yaml      # PTQ method configurations
├── results/
│   ├── tables/                        # CSV results tables
│   └── figures/                       # Charts & visualizations
└── data/                              # Dataset cache (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
```

You'll need a HuggingFace token with access to Gemma models:
```bash
huggingface-cli login
```

## Reproduction

Each notebook is numbered and self-contained. Run them in order on Kaggle (T4 16GB):

1. **01_data_preparation** — Downloads PubMedQA and MedQA, formats for training, runs baseline eval
2. **02_qlora_finetune** — Fine-tunes Gemma 2 2B with QLoRA (PEFT + BnB) on medical QA
3. **03_quantize_and_benchmark** — Applies GPTQ/AWQ/BnB quantization, runs all benchmarks
4. **04_analysis** — Generates comparison tables and figures

## Key Findings

*To be completed after experiments.*

## Hardware

- Fine-tuning & quantization: Kaggle T4 16GB
- Evaluation: Same

## License

MIT
