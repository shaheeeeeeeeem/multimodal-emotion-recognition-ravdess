from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import pandas as pd
import torch
import whisper
from tqdm import tqdm


WHISPER_SAMPLE_RATE = 16000


def transcribe_file(model, audio_path: Path) -> str:
    audio, _ = librosa.load(audio_path, sr=WHISPER_SAMPLE_RATE, mono=True)
    result = model.transcribe(audio, language="en", fp16=False, verbose=False)
    return result["text"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Whisper transcripts for RAVDESS audio clips.")
    parser.add_argument("--index", type=Path, default=Path("data/processed/ravdess_index.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/ravdess"))
    parser.add_argument("--output", type=Path, default=Path("data/transcripts/ravdess_transcripts.csv"))
    parser.add_argument("--model", type=str, default="tiny.en")
    parser.add_argument("--model-cache", type=Path, default=Path("outputs/whisper_cache"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    metadata = pd.read_csv(args.index)
    if args.limit is not None:
        metadata = metadata.head(args.limit).copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model {args.model} on {device}")
    args.model_cache.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(args.model, device=device, download_root=str(args.model_cache))

    rows = []
    for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Transcribing"):
        audio_path = args.dataset_root / row.path
        transcript = transcribe_file(model, audio_path)
        rows.append(
            {
                "path": row.path,
                "filename": row.filename,
                "emotion_id": row.emotion_id,
                "emotion": row.emotion,
                "actor_id": row.actor_id,
                "statement": row.statement,
                "transcript": transcript,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)

    preview_path = args.output.with_suffix(".preview.json")
    with preview_path.open("w", encoding="utf-8") as file:
        json.dump(rows[:10], file, indent=2)

    print(f"Saved {args.output}")
    print(f"Saved {preview_path}")


if __name__ == "__main__":
    main()

