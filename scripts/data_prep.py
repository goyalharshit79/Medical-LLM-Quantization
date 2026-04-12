"""
Data preparation utilities for medical QA datasets.
Downloads PubMedQA and MedQA, formats them for QLoRA fine-tuning with Gemma 4 prompt format.
"""

import json
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets


# --- Gemma 4 Prompt Templates ---

MEDICAL_QA_TEMPLATE = """<|turn>user
{question}<turn|>
<|turn>model
{answer}<turn|>"""

PUBMEDQA_TEMPLATE = """<|turn>user
Context: {context}

Question: {question}
Answer with yes, no, or maybe and explain your reasoning.<turn|>
<|turn>model
{answer}<turn|>"""


def load_pubmedqa():
    """Load PubMedQA dataset (labeled subset). Only has a 'train' split (1000 examples)."""
    ds = load_dataset("pubmed_qa", "pqa_labeled")
    return ds


def load_medqa():
    """Load MedQA (USMLE-style) dataset. Has 'train' and 'test' splits."""
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")
    return ds


def format_pubmedqa_for_training(example):
    """Format a PubMedQA example into Gemma 4 instruction format."""
    # Standard pubmed_qa uses nested context dict with lowercase keys
    ctx_data = example.get("context", {})
    contexts = ctx_data.get("contexts", [])
    context = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
    question = example["question"]

    # Long answer + final decision
    long_answer = example.get("long_answer", "")
    final_decision = example.get("final_decision", "")

    if long_answer and final_decision:
        answer = f"{final_decision}. {long_answer}"
    elif final_decision:
        answer = final_decision
    else:
        answer = long_answer or "No answer available."

    text = PUBMEDQA_TEMPLATE.format(
        context=context,
        question=question,
        answer=answer,
    )
    return {"text": text}


def format_medqa_for_training(example):
    """Format a MedQA example into Gemma 4 instruction format."""
    question = example["question"]
    options = example.get("options", {})
    answer_idx = example.get("answer_idx", "")
    answer = example.get("answer", "")

    # Build options string
    if isinstance(options, dict):
        options_str = "\n".join(f"  {k}) {v}" for k, v in sorted(options.items()))
        full_question = f"{question}\n{options_str}"
    else:
        full_question = question

    if answer_idx and answer:
        full_answer = f"The answer is {answer_idx}) {answer}"
    elif answer:
        full_answer = answer
    else:
        full_answer = "No answer available."

    text = MEDICAL_QA_TEMPLATE.format(
        question=full_question,
        answer=full_answer,
    )
    return {"text": text}


def prepare_calibration_data(num_samples=128, max_length=512, tokenizer=None):
    """
    Prepare medical text for quantization calibration.
    Returns a list of text strings (or tokenized if tokenizer provided).
    Used by GPTQ and AWQ during quantization.
    """
    ds = load_pubmedqa()
    texts = []
    for example in ds["train"]:
        ctx_data = example.get("context", {})
        contexts = ctx_data.get("contexts", [])
        context = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
        texts.append(context)
        if len(texts) >= num_samples:
            break

    if tokenizer is not None:
        # Return tokenized for GPTQ/AWQ calibration
        return tokenizer(texts[:num_samples], truncation=True, max_length=max_length, return_tensors="pt")

    return texts[:num_samples]


def prepare_all_datasets(output_dir="data"):
    """Download and format all datasets, save to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PubMedQA...")
    pubmedqa = load_pubmedqa()  # Only has 'train' split (1000 examples)
    pubmedqa_train = pubmedqa["train"]
    print(f"  Train: {len(pubmedqa_train)} examples (no separate val split)")

    print("Formatting PubMedQA for training...")
    pubmedqa_train_fmt = pubmedqa_train.map(format_pubmedqa_for_training, remove_columns=pubmedqa_train.column_names)

    print("\nLoading MedQA...")
    medqa = load_medqa()  # Has 'train' and 'test' splits
    medqa_train = medqa["train"]
    medqa_test = medqa["test"]

    print(f"  Train: {len(medqa_train)} examples")
    print(f"  Test:  {len(medqa_test)} examples (used as validation)")

    print("Formatting MedQA for training...")
    medqa_train_fmt = medqa_train.map(format_medqa_for_training, remove_columns=medqa_train.column_names)
    medqa_test_fmt = medqa_test.map(format_medqa_for_training, remove_columns=medqa_test.column_names)

    # Combine: PubMedQA train + MedQA train for training
    # Use MedQA test as validation (PubMedQA has no val split)
    train_combined = concatenate_datasets([pubmedqa_train_fmt, medqa_train_fmt])
    val_combined = medqa_test_fmt

    # Save formatted datasets
    train_combined.save_to_disk(str(output_dir / "train"))
    val_combined.save_to_disk(str(output_dir / "val"))

    print(f"\nAll datasets saved to {output_dir}/")
    print(f"  train:    {len(train_combined)} examples (PubMedQA + MedQA)")
    print(f"  val:      {len(val_combined)} examples (PubMedQA + MedQA)")

    return {
        "train": train_combined,
        "val": val_combined,
    }


if __name__ == "__main__":
    prepare_all_datasets()
