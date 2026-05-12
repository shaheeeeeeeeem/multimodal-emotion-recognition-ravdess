from __future__ import annotations

import argparse
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
from src.models.enhanced_audio_cnn import EnhancedAudioCNN


CLASS_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.from_numpy(x).permute(0, 3, 1, 2).float()
    y_tensor = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def class_weights(y_train: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def spec_augment(batch: torch.Tensor, freq_masks: int, time_masks: int, max_freq_width: int, max_time_width: int) -> torch.Tensor:
    augmented = batch.clone()
    _, _, freq_bins, time_bins = augmented.shape

    for item in range(augmented.size(0)):
        for _ in range(freq_masks):
            width = random.randint(0, max_freq_width)
            if width > 0 and width < freq_bins:
                start = random.randint(0, freq_bins - width)
                augmented[item, :, start:start + width, :] = 0.0
        for _ in range(time_masks):
            width = random.randint(0, max_time_width)
            if width > 0 and width < time_bins:
                start = random.randint(0, time_bins - width)
                augmented[item, :, :, start:start + width] = 0.0

    return augmented


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    use_specaugment: bool,
    freq_masks: int,
    time_masks: int,
    max_freq_width: int,
    max_time_width: int,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    predictions = []
    targets = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if is_train and use_specaugment:
            batch_x = spec_augment(batch_x, freq_masks, time_masks, max_freq_width, max_time_width)

        with torch.set_grad_enabled(is_train):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        losses.append(loss.item() * batch_x.size(0))
        predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        targets.extend(batch_y.detach().cpu().numpy().tolist())

    return float(np.sum(losses) / len(targets)), accuracy_score(targets, predictions)


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


def plot_history(history: list[dict[str, float]], output_path: Path, title_prefix: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title(f"{title_prefix} loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="validation")
    axes[1].set_title(f"{title_prefix} accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set_xticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a stronger audio CNN experiment on RAVDESS Mel-spectrograms.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/audio_noise_augmented"))
    parser.add_argument("--run-name", type=str, default="audio_cnn_v2")
    parser.add_argument("--architecture", choices=["baseline", "enhanced"], default="enhanced")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--specaugment", action="store_true")
    parser.add_argument("--freq-masks", type=int, default=1)
    parser.add_argument("--time-masks", type=int, default=1)
    parser.add_argument("--max-freq-width", type=int, default=12)
    parser.add_argument("--max-time-width", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x = np.load(args.data_dir / "x_mel.npy")
    y = np.load(args.data_dir / "y.npy")
    metadata = pd.read_csv(args.data_dir / "metadata_with_splits.csv")
    train_mask = metadata["split"].to_numpy() == "train"
    val_mask = metadata["split"].to_numpy() == "validation"
    test_mask = metadata["split"].to_numpy() == "test"

    train_loader = make_loader(x[train_mask], y[train_mask], args.batch_size, shuffle=True)
    val_loader = make_loader(x[val_mask], y[val_mask], args.batch_size, shuffle=False)
    test_loader = make_loader(x[test_mask], y[test_mask], args.batch_size, shuffle=False)

    if args.architecture == "enhanced":
        model = EnhancedAudioCNN(num_classes=len(CLASS_NAMES)).to(device)
    else:
        model = AudioCNN(num_classes=len(CLASS_NAMES)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights(y[train_mask]).to(device), label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    model_dir = Path("outputs/models")
    metrics_dir = Path("outputs/metrics")
    plots_dir = Path("outputs/plots")
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / f"{args.run_name}_best.pt"

    best = {"loss": float("inf"), "epoch": 0}
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            args.specaugment, args.freq_masks, args.time_masks, args.max_freq_width, args.max_time_width,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device,
            False, args.freq_masks, args.time_masks, args.max_freq_width, args.max_time_width,
        )
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f} | lr={lr:.6f}")

        if val_loss < best["loss"]:
            best = {"loss": val_loss, "epoch": epoch}
            torch.save(model.state_dict(), best_model_path)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    y_true, y_pred, y_probs = predict(model, test_loader, device)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    pd.DataFrame(history).to_csv(metrics_dir / f"{args.run_name}_history.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(metrics_dir / f"{args.run_name}_classification_report.csv")
    pd.DataFrame([{
        "model": args.run_name,
        "architecture": args.architecture,
        "specaugment": args.specaugment,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "best_epoch": best["epoch"],
    }]).to_csv(metrics_dir / f"{args.run_name}_summary.csv", index=False)
    np.save(metrics_dir / f"{args.run_name}_test_probs.npy", y_probs)
    np.save(metrics_dir / f"{args.run_name}_test_true.npy", y_true)
    plot_history(history, plots_dir / f"{args.run_name}_training_curves.png", args.run_name)
    plot_confusion_matrix(matrix, plots_dir / f"{args.run_name}_confusion_matrix.png", f"{args.run_name} confusion matrix")

    print(f"Best validation epoch: {best['epoch']}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"Saved model: {best_model_path}")


if __name__ == "__main__":
    main()
