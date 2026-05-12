from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CLASS_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOTION_TO_INDEX = {emotion: idx for idx, emotion in enumerate(CLASS_NAMES)}


class TranscriptDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(df: pd.DataFrame, tokenizer, max_length: int, batch_size: int, shuffle: bool) -> DataLoader:
    labels = df["emotion"].map(EMOTION_TO_INDEX).to_numpy(dtype=np.int64)
    dataset = TranscriptDataset(df["transcript"].fillna("").astype(str).tolist(), labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader: DataLoader, optimizer, device: torch.device) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    predictions = []
    targets = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(is_train):
            output = model(**batch)
            loss = output.loss
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item() * batch["labels"].size(0))
        predictions.extend(output.logits.argmax(dim=1).detach().cpu().numpy().tolist())
        targets.extend(batch["labels"].detach().cpu().numpy().tolist())

    return float(np.sum(losses) / len(targets)), accuracy_score(targets, predictions)


def predict(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            probs = softmax(output.logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_targets.append(batch["labels"].detach().cpu().numpy())

    return np.concatenate(all_targets), np.concatenate(all_preds), np.concatenate(all_probs)


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("DistilBERT text loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="validation")
    axes[1].set_title("DistilBERT text accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Oranges")
    fig.colorbar(image, ax=axis)
    axis.set_xticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("DistilBERT text confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DistilBERT on Whisper transcripts for the transformer bonus.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/text/metadata_with_splits.csv"))
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--run-name", type=str, default="distilbert_text")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metadata = pd.read_csv(args.metadata)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(CLASS_NAMES)).to(device)

    train_df = metadata[metadata["split"] == "train"].copy()
    val_df = metadata[metadata["split"] == "validation"].copy()
    test_df = metadata[metadata["split"] == "test"].copy()

    train_loader = make_loader(train_df, tokenizer, args.max_length, args.batch_size, shuffle=True)
    val_loader = make_loader(val_df, tokenizer, args.max_length, args.batch_size, shuffle=False)
    test_loader = make_loader(test_df, tokenizer, args.max_length, args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    best = {"loss": float("inf"), "epoch": 0}
    history = []
    model_dir = Path("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / f"{args.run_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_loss < best["loss"]:
            best = {"loss": val_loss, "epoch": epoch}
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    y_true, y_pred, y_probs = predict(model, test_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    metrics_dir = Path("outputs/metrics")
    plots_dir = Path("outputs/plots")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(metrics_dir / f"{args.run_name}_history.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(metrics_dir / f"{args.run_name}_classification_report.csv")
    pd.DataFrame([{"model": args.run_name, "test_accuracy": accuracy, "test_macro_f1": macro_f1, "best_epoch": best["epoch"]}]).to_csv(metrics_dir / f"{args.run_name}_summary.csv", index=False)
    np.save(metrics_dir / f"{args.run_name}_test_probs.npy", y_probs)
    np.save(metrics_dir / f"{args.run_name}_test_true.npy", y_true)
    plot_history(history, plots_dir / f"{args.run_name}_training_curves.png")
    plot_confusion_matrix(matrix, plots_dir / f"{args.run_name}_confusion_matrix.png")

    print(f"Best validation epoch: {best['epoch']}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"Saved model: {best_model_path}")


if __name__ == "__main__":
    main()
