from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one Mel-spectrogram example per emotion.")
    parser.add_argument("--features", type=Path, default=Path("data/processed/audio/x_mel.npy"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/audio/metadata_with_splits.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/mel_spectrogram_examples.png"))
    args = parser.parse_args()

    x = np.load(args.features)
    metadata = pd.read_csv(args.metadata)
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for axis, emotion in zip(axes.ravel(), emotions):
        sample_index = metadata.index[metadata["emotion"] == emotion][0]
        image = x[sample_index, :, :, 0]
        axis.imshow(image, aspect="auto", origin="lower", cmap="magma")
        axis.set_title(emotion)
        axis.set_xlabel("Time frames")
        axis.set_ylabel("Mel bands")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

