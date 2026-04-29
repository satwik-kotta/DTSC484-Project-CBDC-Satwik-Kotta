import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


def train_model(data_path: Path, output_path: Path, test_size: float = 0.2, random_state: int = 42) -> None:
    df = pd.read_csv(data_path)

    if "Sentence" not in df.columns or "You" not in df.columns:
        raise ValueError("CSV must contain 'Sentence' and 'You' columns.")

    df = df[["Sentence", "You"]].dropna()
    df["Sentence"] = df["Sentence"].astype(str)
    df["You"] = pd.to_numeric(df["You"], errors="coerce")
    df = df.dropna()

    X = df["Sentence"]
    y = df["You"].clip(0.0, 1.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    encoder_name = "all-MiniLM-L6-v2"
    encoder = SentenceTransformer(encoder_name)

    X_train_emb = encoder.encode(X_train.tolist(), show_progress_bar=False)
    X_test_emb = encoder.encode(X_test.tolist(), show_progress_bar=False)

    regressor = Ridge(alpha=1.0)
    regressor.fit(X_train_emb, y_train)

    preds = regressor.predict(X_test_emb)
    preds = preds.clip(0.0, 1.0)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    package = {
        "encoder_name": encoder_name,
        "regressor": regressor,
        "text_column": "Sentence",
        "score_column": "You",
        "score_range": [0.0, 1.0],
        "metrics": {
            "mae": float(mae),
            "r2": float(r2),
            "test_size": test_size,
            "random_state": random_state,
            "rows_used": int(len(df)),
        },
    }

    joblib.dump(package, output_path)

    print(f"Saved model to: {output_path}")
    print(f"Rows used: {len(df)}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")


def score_sentence(model_path: Path, text: str) -> None:
    package = joblib.load(model_path)
    encoder = SentenceTransformer(package["encoder_name"])
    regressor = package["regressor"]

    embedding = encoder.encode([text], show_progress_bar=False)
    score = float(regressor.predict(np.asarray(embedding))[0])
    score = max(0.0, min(1.0, score))

    print(f"Sentence: {text}")
    print(f"Predicted score: {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run a sentiment scoring model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model from CSV")
    train_parser.add_argument(
        "--data",
        type=Path,
        default=Path("abhindon.csv"),
        help="Path to CSV file",
    )
    train_parser.add_argument(
        "--out",
        type=Path,
        default=Path("sentiment_model.joblib"),
        help="Where to save trained model",
    )

    score_parser = subparsers.add_parser("score", help="Score a sentence")
    score_parser.add_argument(
        "--model",
        type=Path,
        default=Path("sentiment_model.joblib"),
        help="Path to saved model",
    )
    score_parser.add_argument("--text", type=str, required=True, help="Sentence to score")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.data, args.out)
    elif args.command == "score":
        score_sentence(args.model, args.text)


if __name__ == "__main__":
    main()
