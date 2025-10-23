## Gen-AI Exploration

A collection of experiments exploring Generative AI and LLM fine-tuning techniques.

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

