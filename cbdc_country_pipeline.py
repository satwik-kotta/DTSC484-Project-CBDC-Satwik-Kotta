import os
import pickle
import argparse
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from pypdf import PdfReader
import pytesseract
from PIL import Image


# Keep runtime output clean; this pipeline does not use HF web interface features.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


CBDC_QUERY_TEXTS = [
    "central bank digital currency cbdc digital rupee digital yuan enaira jamdex",
    "digital payments modernization cashless economy mobile wallet adoption",
    "financial inclusion instant payments reduced transaction costs",
    "retail payments government transfers merchant acceptance",
    "secure transparent traceable digital money infrastructure",
]

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
RELEVANCE_THRESHOLD = 0.22
TOP_K = 80
MIN_WORDS = 6
COUNTRY_FOLDERS = {
    "1": ("USA", "/Users/satwik/Documents/newbatch/files of USA"),
    "2": ("Jamaica", "/Users/satwik/Documents/newbatch/files of Jamaica"),
    "3": ("Nigeria", "/Users/satwik/Documents/newbatch/files of Nigeria"),
    "4": ("China", "/Users/satwik/Documents/newbatch/Files of China "),
}


semantic_model = None
query_embedding = None
sentiment_regressor = None


def ensure_nltk_resources() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def load_sentiment_artifact(model_path: Path):
    # Prefer joblib for the existing sentiment_model.joblib artifact.
    try:
        package = joblib.load(model_path)
        return package
    except Exception:
        with open(model_path, "rb") as f:
            return pickle.load(f)


def initialize_models(model_path: Path, semantic_model_name: str) -> None:
    global semantic_model, query_embedding, sentiment_regressor

    semantic_model = SentenceTransformer(semantic_model_name)
    logging.info("Semantic model loaded")

    package = load_sentiment_artifact(model_path)
    if "regressor" not in package:
        raise ValueError("Model artifact is missing 'regressor'. Retrain using sentiment_model.py.")

    sentiment_regressor = package["regressor"]

    query_vectors = semantic_model.encode(CBDC_QUERY_TEXTS, show_progress_bar=False)
    query_embedding = np.mean(query_vectors, axis=0, keepdims=True)


def collect_documents(folder_path: str) -> pd.DataFrame:
    files = []
    for root, _, filenames in os.walk(folder_path):
        for name in filenames:
            if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(Path(root) / name)

    return pd.DataFrame(
        {
            "path": [str(p) for p in files],
            "title": [p.stem for p in files],
        }
    )


def extract_pdf_text(path: str) -> str:
    text = []
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        logging.warning("Failed to open PDF %s: %s", path, exc)
        return ""

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            content = page.extract_text()
            if content:
                text.append(content)
        except Exception as exc:
            logging.warning("PDF text extraction failed for %s page %s: %s", path, page_index, exc)
            continue

    return "\n".join(text)


def extract_image_text(path: str) -> str:
    img = Image.open(path)
    return pytesseract.image_to_string(img)


def extract_text(file_path: str):
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return extract_pdf_text(file_path)
    if ext in {".png", ".jpg", ".jpeg"}:
        return extract_image_text(file_path)
    return None


def get_sentences(text: str):
    return sent_tokenize(text)


def filter_relevant_sentences(sentences):
    cleaned = [s.strip() for s in sentences if len(s.split()) >= MIN_WORDS]

    if not cleaned:
        return []

    embeddings = semantic_model.encode(cleaned, show_progress_bar=False)
    similarities = cosine_similarity(embeddings, query_embedding)[:, 0]

    relevant_idx = np.where(similarities >= RELEVANCE_THRESHOLD)[0]
    if len(relevant_idx) == 0:
        return []

    ranked = sorted(relevant_idx, key=lambda i: similarities[i], reverse=True)
    selected = ranked[:TOP_K]

    return [(cleaned[i], float(similarities[i])) for i in selected]


def score_sentence(sentence: str) -> float:
    embedding = semantic_model.encode([sentence], show_progress_bar=False)
    score = sentiment_regressor.predict(np.asarray(embedding))[0]

    # Clamp between 0 and 1.
    return float(max(0.0, min(1.0, score)))


def score_document(file_path: str):
    text = extract_text(file_path)
    if not text:
        return None, [], 0

    sentences = get_sentences(text)
    relevant = filter_relevant_sentences(sentences)

    if not relevant:
        return None, [], 0

    scores = []
    weights = []
    sentence_rows = []

    for sentence, relevance in relevant:
        sc = score_sentence(sentence)
        weight = len(sentence.split()) * relevance

        scores.append(sc)
        weights.append(weight)
        sentence_rows.append({"sentence": sentence, "score": sc})

    if not scores:
        return None, [], 0

    return float(np.average(scores, weights=weights)), sentence_rows, len(relevant)


def analyze_folder(folder_path: str):
    df = collect_documents(folder_path)
    total_before = len(df)

    df = df.drop_duplicates(subset=["path"], keep="first").reset_index(drop=True)
    removed = total_before - len(df)
    logging.info(
        "Local collection: found %s files in %s",
        total_before,
        folder_path,
    )
    logging.info("After path dedup: %s files (removed %s)", len(df), removed)
    logging.info("Starting sentiment analysis on %s files...", len(df))

    results = []
    sentence_rows = []
    skipped = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        score, rows, relevant_count = score_document(row["path"])
        if score is not None:
            results.append(
                {
                    "title": row["title"],
                    "path": row["path"],
                    "sentiment": score,
                }
            )
            sentence_rows.extend(rows)
            logging.info("[%s/%s] Scored: %.3f (%s relevant sentences)", i, len(df), score, relevant_count)
        else:
            skipped += 1
            logging.info("[%s/%s] Skipped: extraction failed - %s", i, len(df), row["path"])

    logging.info("==================================================")
    logging.info("Extraction Summary:")
    logging.info("  Total files analyzed: %s", len(df))
    logging.info("  Successfully scored: %s", len(results))
    logging.info("  Skipped: %s", skipped)
    logging.info("==================================================")

    return pd.DataFrame(results), pd.DataFrame(sentence_rows), len(df), skipped


def sanitize_filename(value: str) -> str:
    keep = [c if c.isalnum() or c in {"-", "_"} else "_" for c in value.strip()]
    return "".join(keep).strip("_") or "output"


def select_country(country_option: str | None):
    name_to_key = {v[0].lower(): k for k, v in COUNTRY_FOLDERS.items()}

    if country_option is None:
        print("Choose a country:")
        print("1. USA")
        print("2. Jamaica")
        print("3. Nigeria")
        print("4. China")
        country_option = input("Enter 1, 2, 3, or 4: ").strip()

    normalized = country_option.strip()
    if normalized in COUNTRY_FOLDERS:
        return COUNTRY_FOLDERS[normalized]

    key = name_to_key.get(normalized.lower())
    if key:
        return COUNTRY_FOLDERS[key]

    raise ValueError("Invalid country option. Use 1/2/3/4 or USA/Jamaica/Nigeria/China.")


def resolve_country_option(country_option: str | None) -> str:
    if country_option:
        return country_option

    env_country = os.getenv("COUNTRY", "").strip()
    if env_country:
        return env_country

    print("Choose a country:")
    print("1. USA")
    print("2. Jamaica")
    print("3. Nigeria")
    print("4. China")
    print("5. All")
    selected = input("Enter 1, 2, 3, 4, or 5: ").strip()
    if selected == "5":
        return "all"
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CBDC relevance + sentiment pipeline using sentence embeddings and trained regressor."
    )
    parser.add_argument(
        "--country-option",
        type=str,
        default=None,
        help="Country selector: 1=USA, 2=Jamaica, 3=Nigeria, 4=China, or all (also accepts names)",
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default=None,
        help="Optional override for folder containing PDF/image files",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Optional override for country name in output file",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("sentiment_model.joblib"),
        help="Path to trained model artifact from sentiment_model.py",
    )
    parser.add_argument(
        "--semantic-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("."),
        help="Where to save the sentence-level CSV",
    )
    args = parser.parse_args()

    country_option = resolve_country_option(args.country_option)
    country_tasks = []

    normalized = country_option.strip().lower()
    if normalized == "all":
        for _, (country_name, folder_path) in COUNTRY_FOLDERS.items():
            country_tasks.append((country_name, folder_path))
    else:
        selected_country, selected_folder = select_country(country_option)
        country_name = args.country if args.country else selected_country
        input_folder = args.input_folder if args.input_folder else selected_folder
        country_tasks.append((country_name, input_folder))

    ensure_nltk_resources()
    initialize_models(args.model_path, args.semantic_model)

    for country_name, input_folder in country_tasks:
        logging.info("==================================================")
        logging.info("Starting country: %s", country_name)
        logging.info("==================================================")

        doc_df, sentence_df, total_files, skipped_files = analyze_folder(input_folder)

        if doc_df.empty:
            logging.info("No relevant CBDC content found for %s.", country_name)
            continue

        country_score = float(doc_df["sentiment"].mean())
        min_score = float(doc_df["sentiment"].min())
        max_score = float(doc_df["sentiment"].max())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{sanitize_filename(country_name)}_{timestamp}.csv"
        out_path = args.output_folder / out_name

        # Required output format: exactly two columns (sentence, score).
        sentence_df = sentence_df[["sentence", "score"]]
        sentence_df.to_csv(out_path, index=False)

        logging.info("==================================================")
        logging.info("ANALYSIS COMPLETE")
        logging.info("==================================================")
        logging.info("Country Sentiment: %.3f", country_score)
        logging.info("Files analyzed: %s", len(doc_df))
        logging.info("Sentiment range: [%.3f, %.3f]", min_score, max_score)
        logging.info("Dataset saved: %s", out_name)
        logging.info("==================================================")


if __name__ == "__main__":
    main()
