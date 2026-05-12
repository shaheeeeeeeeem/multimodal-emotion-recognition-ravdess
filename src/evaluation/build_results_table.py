from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    metrics_dir = Path("outputs/metrics")
    rows = []

    audio = pd.read_csv(metrics_dir / "audio_cnn_summary.csv")
    text = pd.read_csv(metrics_dir / "text_rnn_summary.csv")
    fusion = pd.read_csv(metrics_dir / "late_fusion_summary.csv")

    rows.append(audio.iloc[0].to_dict())
    rows.append(text.iloc[0].to_dict())
    rows.append(fusion.iloc[0].to_dict())

    table = pd.DataFrame(rows)
    table["test_accuracy_percent"] = (table["test_accuracy"] * 100).round(2)
    table["test_macro_f1_percent"] = (table["test_macro_f1"] * 100).round(2)
    table = table[["model", "test_accuracy", "test_macro_f1", "test_accuracy_percent", "test_macro_f1_percent"]]
    table.to_csv(metrics_dir / "model_comparison.csv", index=False)

    markdown_lines = [
        "| Model | Accuracy (%) | Macro F1 (%) |",
        "|---|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        markdown_lines.append(f"| {row.model} | {row.test_accuracy_percent:.2f} | {row.test_macro_f1_percent:.2f} |")
    markdown = "\n".join(markdown_lines)
    (metrics_dir / "model_comparison.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()

