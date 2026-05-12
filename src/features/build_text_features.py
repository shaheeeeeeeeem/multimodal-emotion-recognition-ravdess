from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


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
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def build_vocab(texts: pd.Series, min_freq: int = 1) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in sorted(counter.items()):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)]
    token_ids = token_ids[:max_length]
    if len(token_ids) < max_length:
        token_ids.extend([vocab[PAD_TOKEN]] * (max_length - len(token_ids)))
    return token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build padded token features from Whisper transcripts.")
    parser.add_argument("--transcripts", type=Path, default=Path("data/transcripts/ravdess_transcripts.csv"))
    parser.add_argument("--audio-metadata", type=Path, default=Path("data/processed/audio/metadata_with_splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/text"))
    parser.add_argument("--max-length", type=int, default=10)
    parser.add_argument("--min-freq", type=int, default=1)
    args = parser.parse_args()

    transcripts = pd.read_csv(args.transcripts)
    audio_metadata = pd.read_csv(args.audio_metadata)[["path", "split"]]
    metadata = transcripts.merge(audio_metadata, on="path", how="left")

    if metadata["split"].isna().any():
        missing = metadata[metadata["split"].isna()]["path"].head().tolist()
        raise ValueError(f"Missing split for transcript rows: {missing}")

    train_texts = metadata.loc[metadata["split"] == "train", "transcript"]
    vocab = build_vocab(train_texts, min_freq=args.min_freq)

    x = np.asarray([encode_text(text, vocab, args.max_length) for text in metadata["transcript"]], dtype=np.int64)
    y = metadata["emotion"].map(EMOTION_TO_INDEX).to_numpy(dtype=np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "x_text.npy", x)
    np.save(args.output_dir / "y.npy", y)
    metadata.to_csv(args.output_dir / "metadata_with_splits.csv", index=False)

    vocab_payload = {
        "vocab": vocab,
        "max_length": args.max_length,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
    }
    with (args.output_dir / "vocab.json").open("w", encoding="utf-8") as file:
        json.dump(vocab_payload, file, indent=2)

    print(f"Saved text features: {args.output_dir / 'x_text.npy'} {x.shape}")
    print(f"Saved labels:        {args.output_dir / 'y.npy'} {y.shape}")
    print(f"Vocabulary size:     {len(vocab)}")
    print(metadata["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
