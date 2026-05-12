from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from build_audio_features import EMOTION_TO_INDEX, audio_to_mel_spectrogram, fix_length, split_by_actor, standardize_from_train


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_clean_audio(audio_path: Path, config: dict) -> np.ndarray:
    sample_rate = config["sample_rate"]
    target_length = int(config["duration_seconds"] * sample_rate)
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=config["trim_top_db"])
    audio = librosa.util.normalize(audio)
    return fix_length(audio, target_length)


def add_background_noise(audio: np.ndarray, noise_factor: float = 0.008) -> np.ndarray:
    noise = np.random.normal(0.0, 1.0, size=audio.shape)
    augmented = audio + noise_factor * noise
    return librosa.util.normalize(augmented)


def shift_pitch(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    shifted = librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=semitones)
    return librosa.util.normalize(shifted)


def mel_from_audio(audio: np.ndarray, config: dict) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build augmented Mel-spectrogram features for RAVDESS.")
    parser.add_argument("--config", type=Path, default=Path("configs/audio_config.json"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/ravdess_index.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/ravdess"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/audio_augmented"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-pitch", action="store_true", help="Also add pitch-up and pitch-down variants. Slower.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    config = load_config(args.config)
    index = pd.read_csv(args.index)
    index = split_by_actor(index, config)

    spectrograms = []
    labels = []
    metadata_rows = []

    for row in tqdm(index.itertuples(index=False), total=len(index), desc="Augmented Mel-spectrograms"):
        audio_path = args.dataset_root / row.path
        label = EMOTION_TO_INDEX[row.emotion]
        base_metadata = row._asdict()

        clean_audio = load_clean_audio(audio_path, config)
        variants = [("clean", clean_audio)]

        if row.split == "train":
            variants.append(("noise", add_background_noise(clean_audio)))
            if args.include_pitch:
                variants.append(("pitch_up", shift_pitch(clean_audio, config["sample_rate"], semitones=1.0)))
                variants.append(("pitch_down", shift_pitch(clean_audio, config["sample_rate"], semitones=-1.0)))

        for augmentation, audio in variants:
            spectrograms.append(mel_from_audio(audio, config))
            labels.append(label)
            item = dict(base_metadata)
            item["augmentation"] = augmentation
            metadata_rows.append(item)

    x = np.stack(spectrograms, axis=0)
    metadata = pd.DataFrame(metadata_rows)
    x = standardize_from_train(x, metadata["split"].to_numpy())
    x = x[..., np.newaxis]
    y = np.asarray(labels, dtype=np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "x_mel.npy", x)
    np.save(args.output_dir / "y.npy", y)
    metadata.to_csv(args.output_dir / "metadata_with_splits.csv", index=False)

    print(f"Saved augmented features: {args.output_dir / 'x_mel.npy'} {x.shape}")
    print(f"Saved labels:             {args.output_dir / 'y.npy'} {y.shape}")
    print(metadata.groupby(["split", "augmentation"]).size().to_string())


if __name__ == "__main__":
    main()

