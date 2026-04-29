DTSC484 — CBDC Sentence-Level Analysis

This repository provides a reproducible pipeline to extract text from PDFs/images, identify CBDC-relevant sentences using embedding-based filtering, and score sentences for a relevance-weighted sentiment estimate using a hybrid of a trained regression model and a BERT assistant.

**Author**: Satwik Kotta

**What this repo contains**:
- Pipeline scripts: `cbdc_country_pipeline.py`, `bert_cbdc_country_pipeline.py` — hybrid relevance + scoring pipeline.
- Model training helpers: `sentiment_model.py` (primary regressor), `bert_sentiment_model.py` (assistant BERT training).
- Example datasets (not included): small CSVs and country folders used during development.
- `requirements.txt` listing Python dependencies.

**High-level pipeline overview**
- Sentence extraction: PDFs are read via `pypdf`; images use `pytesseract` OCR. Sentences are tokenized with NLTK's `punkt`.
- Relevance filtering: a SentenceTransformer semantic model encodes a small list of CBDC query strings; each sentence is embedded and compared against the query embedding. A BERT-based embedding (CLS vectors) is used as an auxiliary similarity signal. The hybrid relevance score = weighted sum of SentenceTransformer similarity and BERT-embedding similarity. Sentences below a configurable threshold (default 0.22) are ignored; the top-K by relevance are retained per document.
- Sentence scoring: retained sentences are scored by a primary regressor (trained on `abhindon.csv`) whose inputs are SentenceTransformer embeddings. An assistant BERT model computes a second score by applying a sigmoid to its logits (model is loaded with `num_labels=1`). The final sentence score is a weighted average: final = (1 - bert_assist_weight) * primary_score + bert_assist_weight * bert_score. Values are clamped to [0,1].

**Key default parameters and where to change them**
- Relevance threshold: `RELEVANCE_THRESHOLD` (default 0.22).
- Top sentences per document: `TOP_K` (default 80).
- Minimum words per sentence considered: `MIN_WORDS` (default 6).
- Relevance weight (ST vs BERT embeddings): `--st-relevance-weight` (default 0.70).
- BERT assistant contribution in final score: `--bert-assist-weight` (default 0.25).

These can be set via CLI flags when running `bert_cbdc_country_pipeline.py`.

**Artifact formats (what the pipeline expects)**
- Primary regressor artifact (`sentiment_model.joblib` or similar): must be a pickled package/dict containing at least the key `regressor` (a scikit-learn-like regressor with `predict`) and optionally `encoder_name` (a SentenceTransformer name). The helper `sentiment_model.py` produces this artifact.
- Assistant BERT artifact (`bert_sentiment_model.joblib`): must include `model_state_dict` and `model_name` keys. Optionally `tokenizer_name` and `max_length` are supported. `bert_sentiment_model.py` produces this artifact.

If you create custom artifacts, ensure these keys exist — the pipeline validates them on load and raises a descriptive error if missing.

**Quick start — recommended layout**
Place data and models in local folders (not tracked in the repo):

```
data/                  # PDF/image folders per country (NOT committed)
models/                # trained artifacts: sentiment_model.joblib, bert_sentiment_model.joblib
results/               # pipeline CSV outputs
```

Example run (BERT-assisted hybrid pipeline):

```bash
source .venv/bin/activate
python bert_cbdc_country_pipeline.py \
	--country-option 1 \
	--input-folder data/files_of_USA \
	--output-folder results \
	--primary-model-path models/sentiment_model.joblib \
	--assistant-bert-model-path models/bert_sentiment_model.joblib \
	--semantic-model all-MiniLM-L6-v2 \
	--st-relevance-weight 0.7 \
	--bert-assist-weight 0.25
```

Notes on flags:
- `--country-option`: accepts `1/2/3/4` for supported country folders or a country name string. Use `all` to run all configured country folders.
- `--input-folder`: override where to read PDF/image files.
- `--primary-model-path` and `--assistant-bert-model-path`: point to your locally stored artifacts.
- `--semantic-model`: Hugging Face / SentenceTransformers model id used for embeddings.

**Training pointers**
- To train the primary regressor (example):

```bash
python sentiment_model.py --train-file data/abhindon.csv --out models/sentiment_model.joblib
```

- To train the assistant BERT model: refer to `bert_sentiment_model.py`; it should produce a package with `model_state_dict` and `model_name`.

Both training scripts include CLI options and basic docstrings — run them with `--help`.

**Troubleshooting & common issues**
- Missing Python packages: create and activate a venv, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- `ModuleNotFoundError: No module named 'joblib'` — install `joblib` via `pip install joblib` or `pip install -r requirements.txt`.
- Tesseract OCR errors: ensure the `tesseract` binary is installed and on `PATH` (Homebrew on macOS: `brew install tesseract`).
- GPU / MPS: the assistant BERT selects `mps` (Apple Silicon) if available, otherwise `cuda` then `cpu`. If loading fails, check `torch` installation and available device support.
- Large models/datasets: keep `data/` and `models/` out of the repo and use external hosting for sharing heavy artifacts.

**Verification checklist**
- Create an isolated virtualenv and install `requirements.txt`.
- Place small sample files in `data/` and the two model artifacts in `models/`.
- Run `bert_cbdc_country_pipeline.py` for a single country and verify `results/` contains a timestamped CSV of sentence-level scores.

**Next steps I can help with**
- Add a minimal runnable notebook that demonstrates running the pipeline on a tiny sample.
- Create `scripts/fetch_data.sh` to download hosted datasets/artifacts.
- Run a smoke test in your environment and fix any runtime errors.

If you want me to commit this README change and push it, say so and I will create the commit for you.

