# LoRA Sentiment Fine-Tuning (DistilBERT)

[![LoRA dry-run validation](https://github.com/FLotfiGit/Gen-AI-Exploration/actions/workflows/validate_lora.yml/badge.svg)](https://github.com/FLotfiGit/Gen-AI-Exploration/actions/workflows/validate_lora.yml)

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
  - `--eval_steps`, `--save_steps`, `--logging_steps_cli` (switches to step-based strategies)
  - `--print_config` (prints effective config and exits), `--dry_run` (show sizes; no training)

## Config files (save/load)

- Load a JSON config (overrides CLI):

```bash
python LoRA_FineTuning_Sentiment.py --config lora_config_example.json
```

- Save the used args to `output_dir/config.json`:

```bash
python LoRA_FineTuning_Sentiment.py --save_config
```

Notes: When `--config` is provided, matching keys overwrite the parsed CLI values. This makes runs reproducible.

`durations.json` also includes an approximate `approx_samples_per_sec` throughput metric.

### Interpreting durations.json

After a run the script writes `durations.json` to the `output_dir`. Typical structure:

```json
{
  "train_seconds": 12.3,
  "eval_seconds": 1.4,
  "approx_train_samples": 1000,
  "approx_samples_per_sec": 81.3
}
```

- `train_seconds` / `eval_seconds`: wall-clock seconds spent in the train/eval call measured locally.
- `approx_train_samples`: a conservative estimate of the number of training samples processed (uses `len(train_dataset) * epochs`, or a 1-step estimate when `--max_steps` is used).
- `approx_samples_per_sec`: `approx_train_samples / train_seconds` when `train_seconds > 0`, otherwise `null`.

Use `approx_samples_per_sec` as a rough indicator of throughput to compare device/configuration changes. It is approximate because it ignores gradient accumulation, loss of time to dataloader/CPU bottlenecks, and possible step limiting via `--max_steps`.

## Predict-only mode

Run inference from a previously saved `output_dir` (no training):

```bash
python LoRA_FineTuning_Sentiment.py \
  --predict_only --output_dir ./lora_sentiment_out \
  --predict_texts "Amazing product!||This was awful."
```

## CI validation workflow

This repository includes a lightweight GitHub Actions workflow that runs a dry-run validation (`tools/validate_lora_dryrun.py`) on pushes and pull requests touching the LoRA script or validator. The workflow installs a minimal set of dependencies and runs the validator to ensure the `--dry_run` code path remains runnable.

You can run the same validation locally:

```bash
python tools/validate_lora_dryrun.py
```

On macOS, if you encounter an OpenMP runtime abort, run:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python tools/validate_lora_dryrun.py
```


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

### macOS OpenMP runtime note

On macOS you may encounter an OpenMP runtime conflict error when importing some numerical libraries (e.g. "Initializing libiomp5.dylib, but found libomp.dylib already initialized"). This is caused by multiple OpenMP runtimes being linked into different native libraries. A safe, conservative approach is to prefer installing packages that don't statically link OpenMP on macOS.

If you need a quick workaround for local experimentation (unsafe for production), set the environment variable KMP_DUPLICATE_LIB_OK=TRUE before running the script:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python LoRA_FineTuning_Sentiment.py --dry_run
```

This bypasses the runtime check but may cause instability in edge cases. Prefer fixing the underlying native library mismatch when possible.
