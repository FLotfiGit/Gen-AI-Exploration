## Gen-AI Exploration

A collection of experiments exploring Generative AI and LLM fine-tuning techniques.

## Testing

This repo includes lightweight smoke tests and validators to ensure the `--dry_run` paths and small demos remain runnable.

- Run the LLaRA and LoRA validators locally:

```bash
python tools/validate_lora_dryrun.py
python tools/validate_llama_demo.py
```

- Run the pytest suite (includes the TF‑IDF and LLaMA dry-run smoke tests):

```bash
python -m pip install pytest
pytest -q
```

CI: The GitHub Actions workflow `validate_lora.yml` runs the validators and `pytest -q` on push/PR so these checks run automatically.

Caching note: CI caches both pip (`~/.cache/pip`) and the Hugging Face hub (`~/.cache/huggingface`) to speed up tokenizer/model fetches. Locally you can clear these with `tools/cleanup_caches.py` (added in this repo) if you need a clean slate.

[![LoRA dry-run validation](https://github.com/FLotfiGit/Gen-AI-Exploration/actions/workflows/validate_lora.yml/badge.svg)](https://github.com/FLotfiGit/Gen-AI-Exploration/actions/workflows/validate_lora.yml)

### Highlights

- LoRA Sentiment Fine-Tuning: See `LoRA_FineTuning_Sentiment.py` and `README_LoRA_Sentiment.md` for a production-ready training script with metrics, early stopping, config management, and optional quantization.
- RAG and TF-IDF: Explore basic RAG and TF-IDF in `Simple_RAG_Conv.py`, `TF_IDF_scratch.py`, and the utilities in `coding_camp.py`.

### TF-IDF mini utilities in `coding_camp.py`

`coding_camp.py` includes a lightweight TF-IDF implementation with sparse cosine similarity and top-k helpers, plus a small demo.

What you can do:
- Fit and transform a small corpus
- Compute cosine similarity on sparse vectors
- Retrieve top-k most similar documents to a query

Try it:
1. Run the built-in demo
	- `python coding_camp.py`
2. Optional smoke test
	- `python -m pytest tests/test_tfidf_basic.py -q` (or run the file directly with Python to execute the assertions)

Note: The optional LoRA demo inside `coding_camp.py` is guarded and will only run if the `peft` and `transformers` libraries are available.

### LLaMA LoRA demo

There is a small, guarded demo `LoRA_Llama_Demo.py` that shows how to attach a LoRA adapter to a causal LM (LLaMA-family). By default it runs in `--dry_run` mode so it won't download very large models during a quick check.

Quick usage:

```bash
# dry run (no downloads)
python LoRA_Llama_Demo.py --dry_run

# actually run generation (may download model and requires transformers/peft)
python LoRA_Llama_Demo.py --run_gen --model_name "meta-llama/Llama-2-7b-chat-hf"
```

Note: Running `--run_gen` requires appropriate resources and dependencies. The repo includes a lightweight validator `tools/validate_llama_demo.py` that runs the demo in `--dry_run` mode for CI or local checks.

### LoRA Experiments (TensorFlow)

The `LoRA_Experiments.py` script is a heavy TensorFlow-based experiment for translation fine-tuning. It defaults to a safe dry-run to avoid large downloads and training:

```bash
# Dry-run (no downloads; quick)
python LoRA_Experiments.py

# Actually run (downloads wmt16, flan-t5-base; requires GPU/TPU and significant time)
python LoRA_Experiments.py --run --max_train 2000 --max_eval 200
```

The script wraps heavy TensorFlow flow in a CLI with logging for transparency. It's useful for exploring LoRA rank and batch-size hyperparameter sweeps.

