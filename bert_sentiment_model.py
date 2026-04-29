import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloader(
    texts: pd.Series,
    labels: pd.Series,
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    encoded = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor(labels.to_numpy(dtype=np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: pd.Series,
    labels: pd.Series,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> tuple[float, float]:
    loader = build_dataloader(texts, labels, tokenizer, max_length, batch_size, shuffle=False)
    predictions = []

    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, _ in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.sigmoid(outputs.logits.squeeze(-1)).detach().cpu().numpy()
            predictions.extend(batch_preds.tolist())

    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return float(mae), float(r2)


def train_model(
    data_path: Path,
    output_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 2e-5,
) -> None:
    set_seed(random_state)

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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    device = get_device()
    model.to(device)

    train_loader = build_dataloader(X_train, y_train, tokenizer, max_length, batch_size, shuffle=True)
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * 0.1)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits.squeeze(-1))
            loss = loss_fn(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs} - train loss: {avg_train_loss:.4f}")

    mae, r2 = evaluate_model(model, tokenizer, X_test, y_test, device, max_length, batch_size)

    package = {
        "model_name": model_name,
        "tokenizer_name": model_name,
        "max_length": max_length,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "text_column": "Sentence",
        "score_column": "You",
        "score_range": [0.0, 1.0],
        "mode": "bert-regression",
        "metrics": {
            "mae": float(mae),
            "r2": float(r2),
            "test_size": test_size,
            "random_state": random_state,
            "rows_used": int(len(df)),
        },
        "model_state_dict": model.state_dict(),
    }

    joblib.dump(package, output_path)

    print(f"Saved model to: {output_path}")
    print(f"Rows used: {len(df)}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")


def score_sentence(model_path: Path, text: str) -> None:
    package = joblib.load(model_path)
    device = get_device()

    if "model_state_dict" in package:
        tokenizer = AutoTokenizer.from_pretrained(package.get("tokenizer_name", package["model_name"]))
        model = AutoModelForSequenceClassification.from_pretrained(package["model_name"], num_labels=1)
        model.load_state_dict(package["model_state_dict"])
        model.to(device)
        model.eval()

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(package.get("max_length", 128)),
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits.squeeze(-1)
            score = float(torch.sigmoid(logits).item())
    else:
        raise ValueError("This model artifact does not contain a BERT regression head.")

    score = max(0.0, min(1.0, score))

    print(f"Sentence: {text}")
    print(f"Predicted score: {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run a BERT sentiment scoring model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train BERT model from CSV")
    train_parser.add_argument("--data", type=Path, default=Path("abhindon.csv"), help="Path to CSV file")
    train_parser.add_argument("--out", type=Path, default=Path("bert_sentiment_model.joblib"), help="Where to save trained model")
    train_parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Pretrained BERT model to fine-tune")
    train_parser.add_argument("--max-length", type=int, default=128, help="Maximum token length for BERT inputs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    train_parser.add_argument("--learning-rate", type=float, default=2e-5, help="Fine-tuning learning rate")

    score_parser = subparsers.add_parser("score", help="Score a sentence")
    score_parser.add_argument("--model", type=Path, default=Path("bert_sentiment_model.joblib"), help="Path to saved model")
    score_parser.add_argument("--text", type=str, required=True, help="Sentence to score")

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            args.data,
            args.out,
            model_name=args.model_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
    elif args.command == "score":
        score_sentence(args.model, args.text)


if __name__ == "__main__":
    main()
