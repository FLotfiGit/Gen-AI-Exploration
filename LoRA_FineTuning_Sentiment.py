#!/usr/bin/env python3
"""
LoRA_FineTuning_Sentiment.py

Clean, runnable example of LoRA fine-tuning for sentiment classification.
- Model: distilbert-base-uncased (sequence classification, 2 labels)
- Dataset: GLUE/SST-2 (with a tiny fallback synthetic dataset if offline)
- Trainer: Hugging Face Trainer with evaluation and simple inference demo

Usage (CPU-friendly quick run):
  python LoRA_FineTuning_Sentiment.py --max_train_samples 800 --max_eval_samples 200 --epochs 1

Notes:
- If the GLUE/SST-2 dataset cannot be downloaded, a small synthetic dataset is used.
- The script auto-detects target LoRA modules for common encoder models (BERT/DistilBERT).
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers import EarlyStoppingCallback
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
try:
    # Optional, used only when quantized loading is requested
    from peft import prepare_model_for_kbit_training  # type: ignore
    _HAS_PREPARE_KBIT = True
except Exception:
    prepare_model_for_kbit_training = None  # type: ignore
    _HAS_PREPARE_KBIT = False


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset helpers
# -----------------------------

def try_load_sst2(max_train: int | None = None, max_eval: int | None = None) -> DatasetDict:
    ds = load_dataset("glue", "sst2")
    if max_train is not None:
        ds["train"] = ds["train"].select(range(min(max_train, len(ds["train"]))))
    if max_eval is not None:
        # Use validation as eval set
        ds["validation"] = ds["validation"].select(range(min(max_eval, len(ds["validation"]))))
    return ds


def make_synthetic_sentiment(n_train: int = 200, n_eval: int = 50) -> DatasetDict:
    pos_phrases = [
        "I love this movie", "This is great", "Absolutely fantastic",
        "What a wonderful experience", "I enjoyed it a lot",
    ]
    neg_phrases = [
        "I hate this", "This is terrible", "Absolutely awful",
        "Worst experience ever", "I didn't like it",
    ]

    def build_split(n: int):
        texts, labels = [], []
        for _ in range(n):
            if random.random() < 0.5:
                texts.append(random.choice(pos_phrases))
                labels.append(1)
            else:
                texts.append(random.choice(neg_phrases))
                labels.append(0)
        return Dataset.from_dict({"sentence": texts, "label": labels})

    return DatasetDict({
        "train": build_split(n_train),
        "validation": build_split(n_eval),
        "test": build_split(n_eval),
    })


# -----------------------------
# LoRA target module detection
# -----------------------------

def detect_lora_targets(model: torch.nn.Module) -> List[str]:
    """Detect likely target modules for LoRA on encoder models.
    Looks for common attention projection names across BERT/DistilBERT/others.
    """
    candidates = ["query", "key", "value", "q_lin", "k_lin", "v_lin", "c_attn"]
    found = set()
    for name, _ in model.named_modules():
        for c in candidates:
            if name.endswith(c) or f".{c}" in name:
                found.add(c)
    # Reasonable defaults if nothing detected
    if not found:
        return ["q_lin", "v_lin"]
    return sorted(found)


# -----------------------------
# Tokenization
# -----------------------------

@dataclass
class TokenizeFn:
    tokenizer: AutoTokenizer
    max_length: int = 128

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.tokenizer(batch["sentence"], truncation=True, max_length=self.max_length)


# -----------------------------
# Metrics
# -----------------------------

def compute_metrics(eval_pred):
    """Compute accuracy and binary F1 without external dependencies."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # Accuracy
    acc = float(np.mean(preds == labels))
    # Binary precision/recall/F1 (positive class = 1)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./lora_sentiment_out")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1, help="If >0, total number of training steps to perform.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="", help="Comma-separated reporters, e.g., wandb,tensorboard; blank disables.")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="", help="Comma-separated LoRA target modules to override auto-detection")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=1000)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge_adapter_and_save", action="store_true", help="Merge LoRA adapter into base weights and save full model")
    parser.add_argument("--save_eval_csv", action="store_true", help="Save evaluation predictions and labels to CSV")
    parser.add_argument("--print_confusion_matrix", action="store_true", help="Print 2x2 confusion matrix after eval")
    parser.add_argument("--eval_on_test", action="store_true", help="Evaluate on test split if available")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load base model in 8-bit with bitsandbytes")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit with bitsandbytes")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load dataset (with fallback)
    try:
        ds = try_load_sst2(args.max_train_samples, args.max_eval_samples)
        using_sst2 = True
        text_col = "sentence"  # SST-2 uses 'sentence'
        label_col = "label"
    except Exception:
        print("Falling back to synthetic dataset (no internet or GLUE unavailable)...")
        ds = make_synthetic_sentiment(args.max_train_samples, args.max_eval_samples)
        using_sst2 = False
        text_col = "sentence"
        label_col = "label"

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model_load_kwargs = {}
    if args.load_in_4bit or args.load_in_8bit:
        # Prefer 4-bit when both are set
        if args.load_in_4bit:
            model_load_kwargs["load_in_4bit"] = True
        elif args.load_in_8bit:
            model_load_kwargs["load_in_8bit"] = True
        model_load_kwargs["device_map"] = "auto"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        **model_load_kwargs,
    )

    # Prepare dataset
    tok_fn = TokenizeFn(tokenizer=tokenizer, max_length=args.max_length)
    ds_tokenized = ds.map(tok_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Attach LoRA
    targets = detect_lora_targets(model)
    parser_targets = None
    # Defer parser access until args exists above; reusing args here
    # Allow overriding target modules from CLI (comma-separated)
    if hasattr(args, 'target_modules'):
        parser_targets = args.target_modules
    else:
        parser_targets = None
    if parser_targets:
        targets = [t.strip() for t in parser_targets.split(',') if t.strip()]
    # If quantized loading requested, prepare model if available
    if (args.load_in_4bit or args.load_in_8bit) and _HAS_PREPARE_KBIT and callable(prepare_model_for_kbit_training):
        model = prepare_model_for_kbit_training(model)
    elif (args.load_in_4bit or args.load_in_8bit) and not _HAS_PREPARE_KBIT:
        print("Warning: prepare_model_for_kbit_training not available; proceeding without special prep.")

    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Training arguments
    report_to = [] if not args.report_to else [x.strip() for x in args.report_to.split(',') if x.strip()]
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        save_total_limit=args.save_total_limit,
        logging_dir=args.logging_dir,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=report_to,  # disable or configure
    )

    # Trainer
    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    eval_name = "validation"
    if args.eval_on_test and "test" in ds_tokenized:
        eval_name = "test"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tokenized["train"],
        eval_dataset=ds_tokenized[eval_name],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Train
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    # Optionally save predictions to CSV and/or print confusion matrix
    if args.save_eval_csv or args.print_confusion_matrix:
        import csv
        eval_dl = trainer.get_eval_dataloader()
        all_logits, all_labels = [], []
        for batch in eval_dl:
            with torch.no_grad():
                logits = trainer.model(**{k: v.to(trainer.model.device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}).logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        preds_np = logits_np.argmax(axis=-1)
        if args.save_eval_csv:
            os.makedirs(args.output_dir, exist_ok=True)
            csv_path = os.path.join(args.output_dir, "eval_predictions.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["label", "pred0", "pred1", "pred_class"])
                for y, logit, pc in zip(labels_np.tolist(), logits_np.tolist(), preds_np.tolist()):
                    w.writerow([int(y), float(logit[0]), float(logit[1]), int(pc)])
            print(f"Saved eval predictions to {csv_path}")
        if args.print_confusion_matrix:
            # rows = true label, cols = predicted label
            cm = np.zeros((2, 2), dtype=int)
            for y, p in zip(labels_np.tolist(), preds_np.tolist()):
                cm[int(y)][int(p)] += 1
            print("Confusion matrix:\n", cm)

    # Save LoRA adapter
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Optionally merge adapter into base model and save full model
    if args.merge_adapter_and_save:
        try:
            merged = model.merge_and_unload()
            merged_dir = os.path.join(args.output_dir, "merged_model")
            os.makedirs(merged_dir, exist_ok=True)
            merged.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"Merged full model saved to {merged_dir}")
        except Exception as e:
            print(f"Merging adapter failed: {e}")

    # Inference demo
    demo_texts = [
        "I absolutely loved this!",
        "This was the worst thing ever.",
        "Not great, but not terrible either.",
    ]
    enc = tokenizer(demo_texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
    for t, p, pr in zip(demo_texts, preds, probs):
        print(f"{t!r} -> pred={int(p)} prob={pr}")


if __name__ == "__main__":
    main()
