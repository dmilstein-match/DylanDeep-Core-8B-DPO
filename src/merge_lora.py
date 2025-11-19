#!/usr/bin/env python3
"""
Merge Abel LoRA adapter with base model for faster vLLM inference.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="GAIR/Abel-7B-002",
        help="Base model identifier",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for merged model",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("LoRA Merge for vLLM Optimization")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Base model: {args.base_model}")
    print(f"  LoRA adapter: {args.lora_path}")
    print(f"  Output path: {args.output_path}\n")

    # Detect dtype
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    dtype_name = "bf16" if use_bf16 else "fp16"

    # Load base model
    print(f"Loading base model in {dtype_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    # Merge LoRA weights into base model
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Save merged model
    print(f"Saving merged model to {args.output_path}...")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print(f"\n{'=' * 80}")
    print("LoRA merge complete!")
    print(f"Merged model saved to: {args.output_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
