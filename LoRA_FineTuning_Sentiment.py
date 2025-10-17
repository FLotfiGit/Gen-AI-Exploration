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
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model


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
        "validation": build_split(n_eval)
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
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=1000)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
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
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Prepare dataset
    tok_fn = TokenizeFn(tokenizer=tokenizer, max_length=args.max_length)
    ds_tokenized = ds.map(tok_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Attach LoRA
    targets = detect_lora_targets(model)
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
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[],  # disable wandb etc.
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tokenized["train"],
        eval_dataset=ds_tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    # Save LoRA adapter
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

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
