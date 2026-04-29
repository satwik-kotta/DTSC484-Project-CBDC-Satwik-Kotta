DTSC484-Project-CBDC-Satwik-Kotta

Project analyzing CBDC (central bank digital currency) mentions and per-sentence sentiment/relevance across country document collections.

## Group Members

- Satwik Kotta

*(Add additional team member full names here if applicable.)*

## Project Summary

This repository contains training scripts and analysis pipelines that:
- Train sentence-level sentiment/regression models from annotated CSVs (`abhindon.csv`).
- Extract text from PDFs and images, identify CBDC-relevant sentences, score sentences with trained models, and export sentence-level CSVs.

Key scripts:
- `sentiment_model.py` — train a SentenceTransformer + Ridge regressor and save `sentiment_model.joblib`.
- `bert_sentiment_model.py` — fine-tune a BERT regression head and save `bert_sentiment_model.joblib`.
- `cbdc_country_pipeline.py` — production-style pipeline using SentenceTransformer + regressor.
- `bert_cbdc_country_pipeline.py` — hybrid pipeline combining SentenceTransformer and BERT assistant.

## Setup Instructions

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Download required external tools (if using images / OCR):
- Install Tesseract OCR on macOS (Homebrew): `brew install tesseract`

4. NLTK punkt resource (the pipelines call `ensure_nltk_resources()` which will download if missing).

## Data Access

- Small sample data files are included in the workspace under the `files of ...` folders.
- Do not commit large raw datasets or model artifacts to GitHub. For the final submission, host large datasets and trained model weights externally (Google Drive, AWS S3, etc.) and place download instructions here.

Example recommended local layout:

```
data/                  # (symlink or downloaded folder - NOT checked in)
  country/             # your full dataset (store externally)
files of USA/          # small sample files included in this workspace
```

To use external data, download it and point pipelines to the folder using `--input-folder`.

## Execution Guide

- Run the primary pipeline for a single country (choose 1=USA,2=Jamaica,3=Nigeria,4=China):

```bash
python cbdc_country_pipeline.py --country-option 1 --model-path sentiment_model.joblib --semantic-model all-MiniLM-L6-v2
```

- Run the hybrid pipeline using the BERT assistant:

```bash
python bert_cbdc_country_pipeline.py --country-option 1 --primary-model-path sentiment_model.joblib --assistant-bert-model-path bert_sentiment_model.joblib
```

- Train the primary regression model (uses `abhindon.csv`):

```bash
python sentiment_model.py train --data abhindon.csv --out sentiment_model.joblib
```

- Train the BERT assistant model:

```bash
python bert_sentiment_model.py train --data abhindon.csv --out bert_sentiment_model.joblib
```

## Important Notes / Best Practices

- Do not commit raw datasets or large model weight files to GitHub. Add them to `.gitignore` and provide external download links.
- `requirements.txt` pins core dependencies used during development.
- Verify the repository on a clean machine before submission: clone, create venv, install, and run one pipeline end-to-end on sample files.

## Contact

If you need me to prepare the remote GitHub repository and push these files, grant me the repository access or run the commands below locally.

## Verification

Before submission, verify on a clean machine:

1. Clone the repo.
2. Create a fresh virtual environment and install `requirements.txt`.
3. Run one training command and one pipeline command on the included sample files to confirm end-to-end functionality.

### Example push commands (using GitHub CLI)

```bash
git init
git add .
git commit -m "Initial commit: README, requirements, gitignore"
# Create a private repo and push (requires GitHub CLI `gh` configured)
gh repo create DTSC484-Project-CBDC-Satwik-Kotta --private --source=. --remote=origin --push
```

If you don't have `gh`, create a private repo via the GitHub web UI and then:

```bash
git remote add origin git@github.com:<your-username>/DTSC484-Project-CBDC-Satwik-Kotta.git
git branch -M main
git push -u origin main
```
