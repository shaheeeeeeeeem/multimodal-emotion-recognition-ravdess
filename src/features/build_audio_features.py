from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


EMOTION_TO_INDEX = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7,
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def fix_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    if len(audio) > target_length:
        return audio[:target_length]
    if len(audio) < target_length:
        padding = target_length - len(audio)
        return np.pad(audio, (0, padding), mode="constant")
    return audio


def audio_to_mel_spectrogram(audio_path: Path, config: dict) -> np.ndarray:
    sample_rate = config["sample_rate"]
    target_length = int(config["duration_seconds"] * sample_rate)

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    audio, _ = librosa.effects.trim(audio, top_db=config["trim_top_db"])
    audio = librosa.util.normalize(audio)
    audio = fix_length(audio, target_length)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db.astype(np.float32)


def split_by_actor(index: pd.DataFrame, config: dict) -> pd.DataFrame:
    actors = np.array(sorted(index["actor_id"].unique()))

    train_val_actors, test_actors = train_test_split(
        actors,
        test_size=config["test_size"],
        random_state=config["random_state"],
    )
    train_actors, validation_actors = train_test_split(
        train_val_actors,
        test_size=config["validation_size"] / (1.0 - config["test_size"]),
        random_state=config["random_state"],
    )

    index = index.copy()
    index["split"] = "train"
    index.loc[index["actor_id"].isin(validation_actors), "split"] = "validation"
    index.loc[index["actor_id"].isin(test_actors), "split"] = "test"
    return index


def standardize_from_train(features: np.ndarray, splits: np.ndarray) -> np.ndarray:
    train_features = features[splits == "train"]
    mean = train_features.mean()
    std = train_features.std()
    return (features - mean) / (std + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mel-spectrogram features for RAVDESS.")
    parser.add_argument("--config", type=Path, default=Path("configs/audio_config.json"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/ravdess_index.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/ravdess"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/audio"))
    args = parser.parse_args()

    config = load_config(args.config)
    index = pd.read_csv(args.index)
    index = split_by_actor(index, config)

    spectrograms = []
    labels = []

    for row in tqdm(index.itertuples(index=False), total=len(index), desc="Mel-spectrograms"):
        audio_path = args.dataset_root / row.path
        spectrograms.append(audio_to_mel_spectrogram(audio_path, config))
        labels.append(EMOTION_TO_INDEX[row.emotion])

    x = np.stack(spectrograms, axis=0)
    x = standardize_from_train(x, index["split"].to_numpy())
    x = x[..., np.newaxis]
    y = np.asarray(labels, dtype=np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "x_mel.npy", x)
    np.save(args.output_dir / "y.npy", y)
    index.to_csv(args.output_dir / "metadata_with_splits.csv", index=False)

    print(f"Saved features: {args.output_dir / 'x_mel.npy'} {x.shape}")
    print(f"Saved labels:   {args.output_dir / 'y.npy'} {y.shape}")
    print(index["split"].value_counts().to_string())


if __name__ == "__main__":
    main()

