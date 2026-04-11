"""
Evaluation utilities: perplexity, downstream task accuracy, and metric logging.
Compatible with Gemma 4 E4B and its prompt format.
"""

import json
import time
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(model, tokenizer, texts, max_length=512, batch_size=4, device="cuda"):
    """
    Compute perplexity over a list of text strings.
    Lower = better language modeling quality.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            mask = encodings["attention_mask"]
            num_tokens = mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def compute_perplexity_wikitext(model, tokenizer, device="cuda", num_samples=512):
    """Compute perplexity on WikiText-2 test set."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:num_samples]
    return compute_perplexity(model, tokenizer, texts, device=device)


def compute_perplexity_medical(model, tokenizer, device="cuda", num_samples=512):
    """Compute perplexity on medical text (PubMedQA contexts)."""
    ds = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", split="test")
    texts = []
    for example in ds:
        ctx = " ".join(example["CONTEXTS"]) if isinstance(example["CONTEXTS"], list) else example["CONTEXTS"]
        if len(ctx.strip()) > 50:
            texts.append(ctx)
        if len(texts) >= num_samples:
            break
    return compute_perplexity(model, tokenizer, texts, device=device)


def evaluate_pubmedqa(model, tokenizer, device="cuda", max_samples=500):
    """
    Evaluate yes/no/maybe accuracy on PubMedQA.
    Uses Gemma 4 prompt format for evaluation.
    """
    ds = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", split="test")

    correct = 0
    total = 0

    for example in tqdm(ds, total=min(len(ds), max_samples), desc="PubMedQA"):
        if total >= max_samples:
            break

        context = " ".join(example["CONTEXTS"]) if isinstance(example["CONTEXTS"], list) else example["CONTEXTS"]
        question = example["QUESTION"]
        gold = example["final_decision"].lower().strip()

        # Gemma 4 prompt format
        prompt = (
            f"<|turn>user\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n"
            f"Answer with exactly one word: yes, no, or maybe.<turn|>\n"
            f"<|turn>model\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        response = response.strip().lower()

        # Extract first word as the answer
        pred = response.split()[0] if response.split() else ""
        pred = pred.strip(".,!?;:")

        if pred in ("yes", "no", "maybe"):
            if pred == gold:
                correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def measure_inference_speed(model, tokenizer, device="cuda", num_runs=20, prompt="What is diabetes?"):
    """Measure tokens/sec for generation."""
    input_text = f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=50, do_sample=False)

    # Timed runs
    total_tokens = 0
    total_time = 0.0

    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += generated_tokens
        total_time += elapsed

    tokens_per_sec = total_tokens / total_time
    return tokens_per_sec


def measure_memory(model, device="cuda"):
    """Measure peak GPU memory usage."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Trigger a forward pass to get realistic memory usage
    dummy = torch.randint(0, 1000, (1, 128), device=device)
    with torch.no_grad():
        model(dummy)

    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    return peak_memory_gb


def get_model_size_gb(model_path):
    """Get model size on disk in GB."""
    path = Path(model_path)
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    elif path.is_file():
        total = path.stat().st_size
    else:
        return 0.0
    return total / (1024**3)


def run_full_evaluation(model, tokenizer, model_name, device="cuda", output_dir="results/tables"):
    """
    Run all evaluations and save results.
    Returns a dict with all metrics.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    results = {"model": model_name}

    # Perplexity
    print("\n[1/5] Computing WikiText-2 perplexity...")
    results["perplexity_wikitext"] = compute_perplexity_wikitext(model, tokenizer, device=device)
    print(f"  -> {results['perplexity_wikitext']:.2f}")

    print("[2/5] Computing medical text perplexity...")
    results["perplexity_medical"] = compute_perplexity_medical(model, tokenizer, device=device)
    print(f"  -> {results['perplexity_medical']:.2f}")

    # Downstream accuracy
    print("[3/5] Evaluating PubMedQA accuracy...")
    pubmedqa = evaluate_pubmedqa(model, tokenizer, device=device)
    results["pubmedqa_accuracy"] = pubmedqa["accuracy"]
    print(f"  -> {pubmedqa['accuracy']:.4f} ({pubmedqa['correct']}/{pubmedqa['total']})")

    # Speed
    print("[4/5] Measuring inference speed...")
    results["tokens_per_sec"] = measure_inference_speed(model, tokenizer, device=device)
    print(f"  -> {results['tokens_per_sec']:.1f} tok/s")

    # Memory
    print("[5/5] Measuring peak VRAM...")
    results["peak_vram_gb"] = measure_memory(model, device=device)
    print(f"  -> {results['peak_vram_gb']:.2f} GB")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "all_results.csv"
    df_new = pd.DataFrame([results])

    if results_file.exists():
        df_existing = pd.read_csv(results_file)
        if model_name in df_existing["model"].values:
            df_existing = df_existing[df_existing["model"] != model_name]
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

    return results
