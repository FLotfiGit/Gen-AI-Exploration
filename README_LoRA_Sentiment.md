# LoRA Sentiment Fine-Tuning (DistilBERT)

This is a clean, runnable example of LoRA fine-tuning for binary sentiment classification using DistilBERT.
It supports GLUE/SST-2 (online) and a synthetic fallback dataset (offline), plus quality-of-life flags for training control.

## Quickstart

CPU-friendly one-epoch run with small samples and CSV export:

```bash
python LoRA_FineTuning_Sentiment.py \
  --epochs 1 --max_train_samples 800 --max_eval_samples 200 \
  --save_eval_csv --print_confusion_matrix
```

## Notable flags

- Dataset and training size
  - `--max_train_samples`, `--max_eval_samples`
  - `--batch_size`, `--epochs`, `--max_steps`
- Training control
  - `--weight_decay`, `--warmup_ratio`, `--gradient_accumulation_steps`
  - `--fp16`, `--bf16`, `--gradient_checkpointing`
  - `--early_stopping`, `--early_stopping_patience`
- LoRA
  - `--r`, `--lora_alpha`, `--lora_dropout`
  - `--target_modules` (override auto-detected modules)
  - `--merge_adapter_and_save` (export merged full model)
- Quantization (optional; requires bitsandbytes/compatible env)
  - `--load_in_8bit`, `--load_in_4bit`
- Evaluation/Logging
  - `--save_eval_csv`, `--eval_on_test`, `--print_confusion_matrix`
  - `--logging_dir`, `--report_to`, `--save_total_limit`

## Example: early stopping and merged export

```bash
python LoRA_FineTuning_Sentiment.py \
  --epochs 3 --early_stopping --early_stopping_patience 2 \
  --merge_adapter_and_save
```

## Outputs

- `output_dir` (default: `./lora_sentiment_out`):
  - LoRA adapter and tokenizer (always saved)
  - `merged_model/` (if `--merge_adapter_and_save`)
  - `eval_predictions.csv` (if `--save_eval_csv`)

## Requirements

See `requirements-lora.txt` for a minimal set of dependencies.

## Tips

- If SST-2 can’t be fetched, the script automatically falls back to a tiny synthetic dataset.
- Use `--target_modules` to precisely control which layers LoRA attaches to.
- For faster training on GPUs, consider `--fp16` or `--bf16` (hardware/driver dependent).
