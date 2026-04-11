# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Systematic evaluation of post-training quantization (PTQ) methods on **Gemma 4 E4B** fine-tuned for medical QA. The research question: *does QLoRA fine-tuning on medical data change which quantization method best preserves model quality?* See `PROJECT_CONTEXT.md` for the full day-by-day plan, budget, and model details — it's the source of truth for project scope and is more detailed than `README.md`.

Target workflow: baseline BF16 → QLoRA fine-tune (Unsloth) → merge → quantize via GPTQ (8/4/3-bit), AWQ (4-bit), BnB-NF4, BnB-INT8 → benchmark all variants on perplexity (WikiText-2 + medical), PubMedQA/MedQA accuracy, VRAM, tokens/sec.

## Architecture

The codebase is small and pipeline-shaped. Four ordered notebooks (`notebooks/01_*` → `04_*`) drive the end-to-end flow; they call into reusable functions in `scripts/`:

- `scripts/data_prep.py` — Loads PubMedQA (`bigbio/pubmed_qa`) and MedQA (`bigbio/med_qa`), formats them with the Gemma 4 prompt template, and also supplies `prepare_calibration_data()` which feeds medical text into GPTQ/AWQ calibration. **Using medical text as calibration data is intentional — it's one of the project's novel angles.**
- `scripts/quantize.py` — One function per PTQ method. GPTQ and AWQ produce saved quantized models; BnB NF4/INT8 are runtime-only (`load_in_4bit` / `load_in_8bit`) and not saved separately. `quantize_all()` runs every variant sequentially with `torch.cuda.empty_cache()` between runs.
- `scripts/evaluate.py` — Metric functions plus `run_full_evaluation()` which appends a row per model variant to `results/tables/all_results.csv` (dedupes by `model` name on re-runs). This CSV is the canonical results store that feeds `04_analysis.ipynb` and eventually the README table.
- `configs/qlora_config.yaml`, `configs/quantization_configs.yaml` — Declarative hyperparameters for fine-tuning and each quantization variant. Keep these in sync with the scripts when adding new variants.

`data/` and `results/` are created at runtime and gitignored.

## Non-obvious constraints

These are easy to get wrong and will silently break things:

- **Model is `google/gemma-4-E4B`** (base, not instruction-tuned). Not Gemma 2, not Gemma 2B. ~4.5B effective params (rest is Per-Layer Embeddings). Native dtype is **bfloat16** — do not use float16.
- **Prompt format is `<|turn>user ... <turn|>` / `<|turn>model ... <turn|>`** — Gemma 4 syntax. This is different from Gemma 2/3's `<start_of_turn>` / `<end_of_turn>`. All training formatting and eval prompts in `data_prep.py` and `evaluate.py` depend on this exact format.
- **Fine-tuning uses Unsloth `FastLanguageModel`**, not vanilla PEFT. LoRA targets all 7 attention+FFN projections (see `qlora_config.yaml`).
- After fine-tuning, LoRA adapters are **merged into the base model** to produce a full BF16 checkpoint — that merged model is the input to every quantization method. Don't quantize an un-merged adapter.
- `evaluate_pubmedqa()` extracts the first word of generation and matches against `{yes, no, maybe}`. If you change the prompt, verify the model still emits a single-word answer first.
- Hardware target: T4 16GB (Colab free) for light work, A100 40GB (Vertex AI) for fine-tuning and GPTQ/AWQ quantization. Budget is ~$50–70 on Vertex AI — avoid running the full `quantize_all()` pipeline speculatively.

## Commands

```bash
pip install -r requirements.txt
huggingface-cli login   # needs Gemma access

# Prepare datasets (writes to data/train, data/val)
python scripts/data_prep.py

# Scripts are library modules — invoke their functions from notebooks or a REPL
# rather than running quantize.py / evaluate.py as __main__.
```

There are no tests, no linter config, and no build step. The notebooks are the runnable entry points.
