# Quantization-Aware Compression of Gemma 4 for Medical QA

A systematic evaluation of post-training quantization (PTQ) methods on a domain-adapted LLM for medical question answering.

## Research Question

> Does QLoRA fine-tuning on medical data change which quantization method best preserves model quality?

## Overview

We fine-tune **Gemma 4 E4B** (~4.5B effective params, April 2026) on medical QA data using QLoRA via [Unsloth](https://github.com/unslothai/unsloth), then systematically evaluate 4 post-training quantization methods across multiple bit-width configurations:

| Method | Bit Widths | Tool |
|--------|-----------|------|
| **GPTQ** | 8, 4, 3 | `auto-gptq` / `transformers` |
| **AWQ** | 4 | `autoawq` |
| **BnB NF4** | 4 | `bitsandbytes` |
| **BnB INT8** | 8 | `bitsandbytes` |

All variants are benchmarked on **PubMedQA** and **MedQA** for downstream accuracy, plus perplexity, memory usage, and inference latency.

## Model

**Gemma 4 E4B** (`google/gemma-4-E4B`) — Google's latest efficient dense model:
- ~8B total parameters, **4.5B effective** (rest is Per-Layer Embeddings)
- 42 decoder layers, hybrid sliding window (512) + global attention
- GQA with 8 query heads / 2 KV heads, 18 KV-shared layers
- 128K token context window, 262K vocabulary
- bfloat16 native dtype

## Results

| Model Variant | Bits | Size (GB) | Perplexity (Wiki) | PubMedQA Acc | MedQA Acc | VRAM (GB) | Tok/s |
|---|---|---|---|---|---|---|---|
| Gemma-4-E4B BF16 (baseline) | 16 | -- | -- | -- | -- | -- | -- |
| Gemma-4-E4B-Med BF16 | 16 | -- | -- | -- | -- | -- | -- |
| Gemma-4-E4B-Med GPTQ-8bit | 8 | -- | -- | -- | -- | -- | -- |
| Gemma-4-E4B-Med GPTQ-4bit | 4 | -- | -- | -- | -- | -- | -- |
| Gemma-4-E4B-Med GPTQ-3bit | 3 | -- | -- | -- | -- | -- | -- |
| Gemma-4-E4B-Med AWQ-4bit | 4 | -- | -- | -- | -- | -- | -- |
| Gemma-4-E4B-Med BnB-NF4 | 4 | -- | -- | -- | -- | -- | -- |
| Gemma-4-E4B-Med BnB-INT8 | 8 | -- | -- | -- | -- | -- | -- |

*Results will be filled as experiments complete.*

## Project Structure

```
.
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Download & format medical QA datasets + baseline eval
│   ├── 02_qlora_finetune.ipynb        # QLoRA fine-tuning via Unsloth
│   ├── 03_quantize_and_benchmark.ipynb # Quantization + full benchmark suite
│   └── 04_analysis.ipynb              # Results analysis & visualization
├── scripts/
│   ├── data_prep.py                   # Dataset loading & formatting utilities
│   ├── evaluate.py                    # Perplexity & accuracy evaluation
│   ├── quantize.py                    # Quantization runners for each method
│   └── benchmark.py                   # Latency, memory, throughput measurement
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

Each notebook is numbered and self-contained. Run them in order:

1. **01_data_preparation** — Downloads PubMedQA and MedQA, formats for training, runs baseline eval
2. **02_qlora_finetune** — Fine-tunes Gemma-4-E4B with QLoRA via Unsloth on medical QA
3. **03_quantize_and_benchmark** — Applies GPTQ/AWQ/BnB quantization, runs all benchmarks
4. **04_analysis** — Generates comparison tables and figures

## Key Findings

*To be completed after experiments.*

## Hardware

- Fine-tuning & quantization: NVIDIA A100 40GB (Vertex AI) / T4 16GB (Colab)
- Evaluation: Same

## License

MIT
