from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.audio_cnn import AudioCNN


CLASS_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x = np.load(data_dir / "x_mel.npy")
    y = np.load(data_dir / "y.npy")
    metadata = pd.read_csv(data_dir / "metadata_with_splits.csv")
    return x, y, metadata


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    # PyTorch CNNs expect channels-first tensors: (batch, channels, height, width).
    x_tensor = torch.from_numpy(x).permute(0, 3, 1, 2).float()
    y_tensor = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def class_weights(y_train: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    predictions = []
    targets = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item() * batch_x.size(0))
        predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        targets.extend(batch_y.detach().cpu().numpy().tolist())

    avg_loss = float(np.sum(losses) / len(targets))
    accuracy = accuracy_score(targets, predictions)
    return avg_loss, accuracy


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            probs = softmax(logits).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_targets.append(batch_y.numpy())

    return np.concatenate(all_targets), np.concatenate(all_preds), np.concatenate(all_probs)


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("Audio CNN loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="validation")
    axes[1].set_title("Audio CNN accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set_xticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Audio CNN confusion matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an audio-only CNN on RAVDESS Mel-spectrograms.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/audio"))
    parser.add_argument("--run-name", type=str, default="audio_cnn")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x, y, metadata = load_arrays(args.data_dir)
    train_mask = metadata["split"].to_numpy() == "train"
    val_mask = metadata["split"].to_numpy() == "validation"
    test_mask = metadata["split"].to_numpy() == "test"

    train_loader = make_loader(x[train_mask], y[train_mask], args.batch_size, shuffle=True)
    val_loader = make_loader(x[val_mask], y[val_mask], args.batch_size, shuffle=False)
    test_loader = make_loader(x[test_mask], y[test_mask], args.batch_size, shuffle=False)

    model = AudioCNN(num_classes=len(CLASS_NAMES)).to(device)
    weights = class_weights(y[train_mask]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_fingerprint = {"loss": float("inf"), "epoch": 0}
    history = []
    model_dir = Path("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / f"{args.run_name}_best.pt"

    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f} | lr={current_lr:.6f}")

        if val_loss < best_val_fingerprint["loss"]:
            best_val_fingerprint = {"loss": val_loss, "epoch": epoch}
            torch.save(model.state_dict(), best_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    y_true, y_pred, y_probs = predict(model, test_loader, device)

    metrics_dir = Path("outputs/metrics")
    plots_dir = Path("outputs/plots")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    pd.DataFrame(history).to_csv(metrics_dir / f"{args.run_name}_history.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(metrics_dir / f"{args.run_name}_classification_report.csv")
    pd.DataFrame([{"model": args.run_name, "test_accuracy": accuracy, "test_macro_f1": macro_f1, "best_epoch": best_val_fingerprint["epoch"]}]).to_csv(metrics_dir / f"{args.run_name}_summary.csv", index=False)
    np.save(metrics_dir / f"{args.run_name}_test_probs.npy", y_probs)
    np.save(metrics_dir / f"{args.run_name}_test_true.npy", y_true)

    plot_history(history, plots_dir / f"{args.run_name}_training_curves.png")
    plot_confusion_matrix(matrix, plots_dir / f"{args.run_name}_confusion_matrix.png")

    print(f"Best validation epoch: {best_val_fingerprint['epoch']}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"Saved model: {best_model_path}")


if __name__ == "__main__":
    main()

