#!/usr/bin/env python3
"""LoRA_Llama_Demo.py

Tiny, guarded demonstration of attaching a LoRA adapter to a causal LM (LLaMA-family).

This script intentionally defaults to a safe `--dry_run` behavior so CI or local checks
don't attempt to download very large models. To actually run generation you must pass
`--run_gen` and have appropriate model weights and dependencies installed.

Examples:
  # quick check (no downloads)
  python LoRA_Llama_Demo.py --dry_run

  # actually run (may download large model and requires transformers/peft)
  python LoRA_Llama_Demo.py --run_gen --model_name "meta-llama/Llama-2-7b-chat-hf"

"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--dry_run", action="store_true", help="Perform a dry-run (no downloads).")
    parser.add_argument("--run_gen", action="store_true", help="Actually load model and run a tiny generation (requires deps).")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank for demo")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] LLaMA LoRA demo would attach LoRA(r={}) to model '{}'.".format(args.r, args.model_name))
        print("To actually run generation, call with --run_gen and ensure transformers/peft are installed.")
        return 0

    # Only attempt real work if explicitly requested
    if not args.run_gen:
        print("No action requested. Use --dry_run or --run_gen.")
        return 0

    # Try to import transformers and peft dynamically so this file can be imported safely
    try:
        import importlib
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
    except Exception as e:
        print("Required libraries not available (transformers/peft). Install them to run generation:")
        print("  pip install transformers peft")
        print("Error:", e)
        return 2

    # Perform a minimal attach + generation demo
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        model_name = args.model_name
        print(f"Loading tokenizer and model: {model_name} (this may download large files)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Attach a tiny LoRA adapter (demo only)
        lora_cfg = LoraConfig(r=args.r, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)
        print("LoRA adapter attached. Running a short generation...")

        prompt = "Translate to French: Hello world.\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        gen = model.generate(**inputs, max_new_tokens=20)
        out = tokenizer.decode(gen[0], skip_special_tokens=True)
        print("Generation result:\n", out)
        return 0
    except Exception as e:
        print("Demo failed during execution:", e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
