"""
Quantization runners for GPTQ, AWQ, and BitsAndBytes methods.
Each function takes a model path and returns a quantized model.
Compatible with Gemma 4 E4B architecture.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig

from data_prep import prepare_calibration_data


def quantize_gptq(model_path, bits=4, group_size=128, output_dir=None, num_calibration_samples=128):
    """
    Quantize a model using GPTQ.
    Requires: auto-gptq

    Args:
        model_path: Path to the BF16 model (merged fine-tuned model)
        bits: Quantization bit width (3, 4, or 8)
        group_size: Group size for quantization
        output_dir: Where to save the quantized model
        num_calibration_samples: Number of calibration examples
    """
    print(f"\n--- GPTQ {bits}-bit quantization ---")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Prepare calibration dataset (medical text from PubMedQA)
    print("Preparing medical calibration data...")
    calibration_texts = prepare_calibration_data(
        num_samples=num_calibration_samples,
        tokenizer=None,  # Return raw text
    )

    # Configure GPTQ via transformers
    gptq_config = GPTQConfig(
        bits=bits,
        group_size=group_size,
        desc_act=True,
        dataset=calibration_texts,
        tokenizer=tokenizer,
    )

    print(f"Loading model and quantizing to {bits}-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=gptq_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"Saved GPTQ-{bits}bit model to {output_dir}")

    return model, tokenizer


def quantize_awq(model_path, w_bit=4, q_group_size=128, output_dir=None):
    """
    Quantize a model using AWQ.
    Requires: autoawq

    Args:
        model_path: Path to the BF16 model
        w_bit: Weight bit width (typically 4)
        q_group_size: Quantization group size
        output_dir: Where to save the quantized model
    """
    print(f"\n--- AWQ {w_bit}-bit quantization ---")

    from awq import AutoAWQForCausalLM

    print("Loading model for AWQ quantization...")
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Prepare calibration data (medical text)
    calibration_texts = prepare_calibration_data(num_samples=128, tokenizer=None)

    quant_config = {
        "zero_point": True,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": "GEMM",
    }

    print(f"Quantizing to {w_bit}-bit with AWQ...")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_texts)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_quantized(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"Saved AWQ-{w_bit}bit model to {output_dir}")

    return model, tokenizer


def load_bnb_nf4(model_path):
    """
    Load model with BitsAndBytes NF4 quantization.
    This is runtime quantization -- no separate quantized model is saved.

    Args:
        model_path: Path to the BF16 model
    """
    print("\n--- BitsAndBytes NF4 (4-bit) ---")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Loaded model in NF4 4-bit")
    return model, tokenizer


def load_bnb_int8(model_path):
    """
    Load model with BitsAndBytes INT8 quantization.
    Uses mixed-precision decomposition for outlier features.

    Args:
        model_path: Path to the BF16 model
    """
    print("\n--- BitsAndBytes INT8 (8-bit) ---")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Loaded model in INT8 8-bit")
    return model, tokenizer


def quantize_all(model_path, output_base_dir="models/quantized"):
    """
    Run all quantization methods on a given model.
    Returns dict of {method_name: (model, tokenizer)}.
    """
    output_base = Path(output_base_dir)
    results = {}

    # GPTQ variants
    for bits in [8, 4, 3]:
        name = f"gptq-{bits}bit"
        model, tokenizer = quantize_gptq(
            model_path, bits=bits, output_dir=output_base / name
        )
        results[name] = (model, tokenizer)
        del model
        torch.cuda.empty_cache()

    # AWQ
    name = "awq-4bit"
    model, tokenizer = quantize_awq(
        model_path, output_dir=output_base / name
    )
    results[name] = (model, tokenizer)
    del model
    torch.cuda.empty_cache()

    # BnB variants (runtime, no save)
    name = "bnb-nf4"
    model, tokenizer = load_bnb_nf4(model_path)
    results[name] = (model, tokenizer)
    del model
    torch.cuda.empty_cache()

    name = "bnb-int8"
    model, tokenizer = load_bnb_int8(model_path)
    results[name] = (model, tokenizer)
    del model
    torch.cuda.empty_cache()

    return results
