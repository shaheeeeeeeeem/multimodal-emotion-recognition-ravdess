from __future__ import annotations

import argparse
import csv
from pathlib import Path


EMOTION_LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITY_LABELS = {
    "01": "normal",
    "02": "strong",
}

STATEMENT_LABELS = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


def parse_ravdess_filename(path: Path, dataset_root: Path) -> dict[str, str | int]:
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS filename format: {path.name}")

    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts

    return {
        "path": str(path.relative_to(dataset_root).as_posix()),
        "filename": path.name,
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion_id": int(emotion),
        "emotion_code": emotion,
        "emotion": EMOTION_LABELS[emotion],
        "intensity_id": int(intensity),
        "intensity": INTENSITY_LABELS[intensity],
        "statement_id": int(statement),
        "statement": STATEMENT_LABELS[statement],
        "repetition_id": int(repetition),
        "actor_id": int(actor),
        "gender": "female" if int(actor) % 2 == 0 else "male",
    }


def find_audio_files(dataset_root: Path) -> list[Path]:
    actor_dirs = sorted(
        path
        for path in dataset_root.glob("Actor_*")
        if path.is_dir() and path.name != "audio_speech_actors_01-24"
    )
    audio_files: list[Path] = []
    for actor_dir in actor_dirs:
        audio_files.extend(sorted(actor_dir.glob("*.wav")))
    return audio_files


def write_index(rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "filename",
        "modality",
        "vocal_channel",
        "emotion_id",
        "emotion_code",
        "emotion",
        "intensity_id",
        "intensity",
        "statement_id",
        "statement",
        "repetition_id",
        "actor_id",
        "gender",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean metadata index for RAVDESS audio files.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/ravdess"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/ravdess_index.csv"))
    args = parser.parse_args()

    audio_files = find_audio_files(args.dataset_root)
    rows = [parse_ravdess_filename(path, args.dataset_root) for path in audio_files]
    write_index(rows, args.output)

    print(f"Indexed {len(rows)} audio files")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

