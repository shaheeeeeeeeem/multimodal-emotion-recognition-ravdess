from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


CLASS_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def evaluate(name: str, y_true: np.ndarray, probs: np.ndarray) -> dict[str, float | str]:
    y_pred = probs.argmax(axis=1)
    return {
        "model": name,
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def max_confidence(audio_probs: np.ndarray, text_probs: np.ndarray) -> np.ndarray:
    audio_conf = audio_probs.max(axis=1)
    text_conf = text_probs.max(axis=1)
    use_audio = audio_conf >= text_conf
    fused = text_probs.copy()
    fused[use_audio] = audio_probs[use_audio]
    return fused


def plot_confusion_matrix(matrix: np.ndarray, title: str, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Purples")
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
    parser = argparse.ArgumentParser(description="Evaluate late fusion strategies for audio and text emotion models.")
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()

    audio_probs = np.load(args.metrics_dir / "audio_cnn_test_probs.npy")
    text_probs = np.load(args.metrics_dir / "text_rnn_test_probs.npy")
    audio_true = np.load(args.metrics_dir / "audio_cnn_test_true.npy")
    text_true = np.load(args.metrics_dir / "text_rnn_test_true.npy")

    if not np.array_equal(audio_true, text_true):
        raise ValueError("Audio and text test labels are not aligned. Fusion would be invalid.")

    y_true = audio_true
    strategies = {
        "late_fusion_average": (audio_probs + text_probs) / 2.0,
        "late_fusion_weighted_70_audio_30_text": 0.7 * audio_probs + 0.3 * text_probs,
        "late_fusion_weighted_85_audio_15_text": 0.85 * audio_probs + 0.15 * text_probs,
        "late_fusion_max_confidence": max_confidence(audio_probs, text_probs),
    }

    rows = []
    reports = {}
    matrices = {}
    for name, probs in strategies.items():
        rows.append(evaluate(name, y_true, probs))
        y_pred = probs.argmax(axis=1)
        reports[name] = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        matrices[name] = confusion_matrix(y_true, y_pred)

    summary = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.metrics_dir / "late_fusion_summary.csv", index=False)

    best_name = str(summary.iloc[0]["model"])
    pd.DataFrame(reports[best_name]).transpose().to_csv(args.metrics_dir / "late_fusion_best_classification_report.csv")
    plot_confusion_matrix(
        matrices[best_name],
        f"Best late fusion confusion matrix: {best_name}",
        args.plots_dir / "late_fusion_best_confusion_matrix.png",
    )

    print(summary.to_string(index=False))
    print(f"Best strategy: {best_name}")


if __name__ == "__main__":
    main()
