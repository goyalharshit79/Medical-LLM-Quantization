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


def load_pubmedqa(split="train"):
    """Load PubMedQA dataset (labeled subset)."""
    ds = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", split=split)
    return ds


def load_medqa(split="train"):
    """Load MedQA (USMLE-style) dataset."""
    ds = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split=split)
    return ds


def format_pubmedqa_for_training(example):
    """Format a PubMedQA example into Gemma 4 instruction format."""
    context = " ".join(example["CONTEXTS"]) if isinstance(example["CONTEXTS"], list) else example["CONTEXTS"]
    question = example["QUESTION"]

    # Long answer + final decision
    long_answer = example.get("LONG_ANSWER", "")
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
    ds = load_pubmedqa("train")
    texts = []
    for example in ds:
        context = " ".join(example["CONTEXTS"]) if isinstance(example["CONTEXTS"], list) else example["CONTEXTS"]
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
    pubmedqa_train = load_pubmedqa("train")
    pubmedqa_val = load_pubmedqa("validation")

    print(f"  Train: {len(pubmedqa_train)} examples")
    print(f"  Val:   {len(pubmedqa_val)} examples")

    print("Formatting PubMedQA for training...")
    pubmedqa_train_fmt = pubmedqa_train.map(format_pubmedqa_for_training, remove_columns=pubmedqa_train.column_names)
    pubmedqa_val_fmt = pubmedqa_val.map(format_pubmedqa_for_training, remove_columns=pubmedqa_val.column_names)

    print("\nLoading MedQA...")
    medqa_train = load_medqa("train")
    medqa_val = load_medqa("validation")

    print(f"  Train: {len(medqa_train)} examples")
    print(f"  Val:   {len(medqa_val)} examples")

    print("Formatting MedQA for training...")
    medqa_train_fmt = medqa_train.map(format_medqa_for_training, remove_columns=medqa_train.column_names)
    medqa_val_fmt = medqa_val.map(format_medqa_for_training, remove_columns=medqa_val.column_names)

    # Combine into single train/val
    train_combined = concatenate_datasets([pubmedqa_train_fmt, medqa_train_fmt])
    val_combined = concatenate_datasets([pubmedqa_val_fmt, medqa_val_fmt])

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
