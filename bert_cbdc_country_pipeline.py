import os
import pickle
import argparse
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import nltk
from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pypdf import PdfReader
import pytesseract
from PIL import Image


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
# Main relevance is SentenceTransformer, with BERT as contextual assistant.
ST_RELEVANCE_WEIGHT = 0.70
# Main score is the abhindon-trained model, with BERT as assistant only.
BERT_ASSIST_WEIGHT = 0.25
COUNTRY_FOLDERS = {
    "1": ("USA", "/Users/satwik/Documents/newbatch/files of USA"),
    "2": ("Jamaica", "/Users/satwik/Documents/newbatch/files of Jamaica"),
    "3": ("Nigeria", "/Users/satwik/Documents/newbatch/files of Nigeria"),
    "4": ("China", "/Users/satwik/Documents/newbatch/Files of China "),
}


semantic_model = None
primary_query_embedding = None
bert_query_embedding = None

primary_sentiment_encoder = None
primary_sentiment_regressor = None

bert_sentiment_model = None
bert_sentiment_tokenizer = None
bert_sentiment_device = None
bert_sentiment_max_length = 128

st_relevance_weight = ST_RELEVANCE_WEIGHT
bert_assist_weight = BERT_ASSIST_WEIGHT


def ensure_nltk_resources() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def load_sentiment_artifact(model_path: Path):
    try:
        package = joblib.load(model_path)
        return package
    except Exception:
        with open(model_path, "rb") as f:
            return pickle.load(f)


def _bert_cls_embeddings(texts: list[str], batch_size: int = 16) -> np.ndarray:
    all_vectors = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = bert_sentiment_tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=bert_sentiment_max_length,
            padding=True,
        )
        encoded = {key: value.to(bert_sentiment_device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = bert_sentiment_model(**encoded, output_hidden_states=True, return_dict=True)
            cls_vectors = outputs.hidden_states[-1][:, 0, :].detach().cpu().numpy()

        all_vectors.append(cls_vectors)

    vectors = np.vstack(all_vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def initialize_models(primary_model_path: Path, assistant_model_path: Path, semantic_model_name: str) -> None:
    global semantic_model, primary_query_embedding, bert_query_embedding
    global primary_sentiment_encoder, primary_sentiment_regressor
    global bert_sentiment_model, bert_sentiment_tokenizer, bert_sentiment_device, bert_sentiment_max_length

    semantic_model = SentenceTransformer(semantic_model_name)
    logging.info("Semantic model loaded")

    primary_package = load_sentiment_artifact(primary_model_path)
    if "regressor" not in primary_package:
        raise ValueError("Primary model artifact is missing 'regressor'. Train using sentiment_model.py with abhindon.csv.")

    primary_encoder_name = primary_package.get("encoder_name", semantic_model_name)
    primary_sentiment_encoder = SentenceTransformer(primary_encoder_name)
    primary_sentiment_regressor = primary_package["regressor"]
    logging.info("Primary sentiment model loaded (from abhindon-trained artifact)")

    assistant_package = load_sentiment_artifact(assistant_model_path)
    if "model_state_dict" not in assistant_package:
        raise ValueError("Assistant BERT artifact is missing 'model_state_dict'. Train using bert_sentiment_model.py.")

    bert_sentiment_max_length = int(assistant_package.get("max_length", 128))
    bert_sentiment_device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    bert_sentiment_tokenizer = AutoTokenizer.from_pretrained(
        assistant_package.get("tokenizer_name", assistant_package["model_name"])
    )
    bert_sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        assistant_package["model_name"], num_labels=1
    )
    bert_sentiment_model.load_state_dict(assistant_package["model_state_dict"])
    bert_sentiment_model.to(bert_sentiment_device)
    bert_sentiment_model.eval()
    logging.info("Assistant BERT model loaded: %s", assistant_package["model_name"])

    query_vectors = semantic_model.encode(CBDC_QUERY_TEXTS, show_progress_bar=False)
    primary_query_embedding = np.mean(query_vectors, axis=0, keepdims=True)

    bert_query_vectors = _bert_cls_embeddings(CBDC_QUERY_TEXTS)
    bert_query_embedding = np.mean(bert_query_vectors, axis=0, keepdims=True)


def collect_documents(folder_path: str) -> pd.DataFrame:
    files = []
    for root, _, filenames in os.walk(folder_path):
        for name in filenames:
            if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(Path(root) / name)

    return pd.DataFrame({"path": [str(p) for p in files], "title": [p.stem for p in files]})


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

    st_embeddings = semantic_model.encode(cleaned, show_progress_bar=False)
    st_similarities = cosine_similarity(st_embeddings, primary_query_embedding)[:, 0]

    bert_embeddings = _bert_cls_embeddings(cleaned)
    bert_similarities = cosine_similarity(bert_embeddings, bert_query_embedding)[:, 0]

    similarities = (st_relevance_weight * st_similarities) + ((1.0 - st_relevance_weight) * bert_similarities)

    relevant_idx = np.where(similarities >= RELEVANCE_THRESHOLD)[0]
    if len(relevant_idx) == 0:
        return []

    ranked = sorted(relevant_idx, key=lambda i: similarities[i], reverse=True)
    selected = ranked[:TOP_K]
    return [(cleaned[i], float(similarities[i])) for i in selected]


def _bert_assistant_score(sentence: str) -> float:
    encoded = bert_sentiment_tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=bert_sentiment_max_length,
        padding=True,
    )
    encoded = {key: value.to(bert_sentiment_device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = bert_sentiment_model(**encoded).logits.squeeze(-1)
        score = float(torch.sigmoid(logits).item())

    return float(max(0.0, min(1.0, score)))


def score_sentence(sentence: str) -> float:
    primary_embedding = primary_sentiment_encoder.encode([sentence], show_progress_bar=False)
    primary_score = float(primary_sentiment_regressor.predict(np.asarray(primary_embedding))[0])
    primary_score = float(max(0.0, min(1.0, primary_score)))

    bert_score = _bert_assistant_score(sentence)
    score = ((1.0 - bert_assist_weight) * primary_score) + (bert_assist_weight * bert_score)

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
    logging.info("Local collection: found %s files in %s", total_before, folder_path)
    logging.info("After path dedup: %s files (removed %s)", len(df), removed)
    logging.info("Starting sentiment analysis on %s files...", len(df))

    results = []
    sentence_rows = []
    skipped = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        score, rows, relevant_count = score_document(row["path"])
        if score is not None:
            results.append({"title": row["title"], "path": row["path"], "sentiment": score})
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
    global st_relevance_weight, bert_assist_weight

    parser = argparse.ArgumentParser(
        description="Hybrid CBDC pipeline: primary abhindon-trained model + BERT contextual assistance."
    )
    parser.add_argument("--country-option", type=str, default=None, help="Country selector: 1=USA, 2=Jamaica, 3=Nigeria, 4=China, or all (also accepts names)")
    parser.add_argument("--input-folder", type=str, default=None, help="Optional override for folder containing PDF/image files")
    parser.add_argument("--country", type=str, default=None, help="Optional override for country name in output file")
    parser.add_argument(
        "--primary-model-path",
        type=Path,
        default=Path("sentiment_model.joblib"),
        help="Path to primary sentiment model artifact from sentiment_model.py (trained on abhindon.csv)",
    )
    parser.add_argument(
        "--assistant-bert-model-path",
        type=Path,
        default=Path("bert_sentiment_model.joblib"),
        help="Path to assistant BERT model artifact from bert_sentiment_model.py",
    )
    parser.add_argument("--semantic-model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument(
        "--st-relevance-weight",
        type=float,
        default=ST_RELEVANCE_WEIGHT,
        help="Weight for SentenceTransformer relevance in hybrid relevance score (0 to 1)",
    )
    parser.add_argument(
        "--bert-assist-weight",
        type=float,
        default=BERT_ASSIST_WEIGHT,
        help="Assistant BERT contribution to final sentence score (0 to 1). Primary model remains dominant.",
    )
    parser.add_argument("--output-folder", type=Path, default=Path("."), help="Where to save the sentence-level CSV")
    args = parser.parse_args()

    if not (0.0 <= args.st_relevance_weight <= 1.0):
        raise ValueError("--st-relevance-weight must be between 0 and 1")
    if not (0.0 <= args.bert_assist_weight <= 1.0):
        raise ValueError("--bert-assist-weight must be between 0 and 1")

    st_relevance_weight = args.st_relevance_weight
    bert_assist_weight = args.bert_assist_weight

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
    initialize_models(args.primary_model_path, args.assistant_bert_model_path, args.semantic_model)

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
