# Multimodal Emotion Recognition on RAVDESS

This repository implements a multimodal emotion-recognition pipeline for the RAVDESS Emotional Speech dataset. The project classifies one of 8 emotions using:

- Audio Mel-spectrograms with a PyTorch CNN
- Whisper-generated transcripts with GRU and DistilBERT text models
- Late-fusion probability averaging across audio and text branches

The best result came from the bonus experiment: noise-augmented audio CNN fused with DistilBERT text probabilities.

## Results

| Model | Accuracy (%) | Macro F1 (%) | Notes |
|---|---:|---:|---|
| Audio CNN | 38.33 | 34.24 | Baseline Mel-spectrogram CNN |
| Text GRU | 16.25 | 9.67 | Whisper transcript RNN branch |
| Late Fusion Average | 39.17 | 35.30 | Baseline audio + text probability averaging |
| Audio CNN + Noise Augmentation | 46.67 | 43.10 | Bonus augmentation experiment |
| DistilBERT Text Branch | 17.50 | 8.01 | Bonus transformer text branch |
| Augmented Audio + DistilBERT Fusion | 47.92 | 44.38 | Best overall result |

## Why Accuracy Is Not Extremely High

The project uses an actor-based split: speakers in the test set are not present in the training set. This is harder than a random clip split, but more honest because the model must generalize to unseen voices instead of recognizing the same actor's voice style.

The text branch is also naturally weak on RAVDESS because most clips contain only two fixed sentences:

```text
Kids are talking by the door.
Dogs are sitting by the door.
```

That means emotion is mostly carried by tone, pitch, rhythm, and intensity rather than word meaning.

## Dataset

Dataset: RAVDESS Emotional Speech Audio from Kaggle.

Expected local layout after downloading and extracting:

```text
data/raw/ravdess/Actor_01/03-01-01-01-01-01-01.wav
data/raw/ravdess/Actor_02/03-01-03-02-02-01-02.wav
```

The raw dataset is not committed to this repository because it is large and should be downloaded from Kaggle.

Emotion labels are encoded in the third filename field:

| Code | Emotion |
|---|---|
| 01 | neutral |
| 02 | calm |
| 03 | happy |
| 04 | sad |
| 05 | angry |
| 06 | fearful |
| 07 | disgust |
| 08 | surprised |

## Project Structure

```text
configs/                 Experiment configuration
src/data/                Dataset indexing and filename label parsing
src/features/            Audio, transcript, and augmentation feature builders
src/models/              PyTorch model definitions
src/training/            CNN, GRU, and DistilBERT training scripts
src/fusion/              Late-fusion evaluation
src/evaluation/          Result-table generation
reports/                 Technical report
reports/figures/         Report figures
outputs/metrics/         Saved result CSV files
outputs/plots/           Training curves and confusion matrices
```

## Methods

### Audio CNN

Audio clips are converted into Mel-spectrograms using `librosa`. The CNN treats each spectrogram as a grayscale image and learns time-frequency patterns associated with emotion.

Preprocessing steps:

1. Load audio as mono.
2. Resample to 22,050 Hz.
3. Trim silence.
4. Normalize volume.
5. Pad or crop to 3.5 seconds.
6. Convert to Mel-spectrogram.
7. Convert power to decibels.
8. Standardize using training-set statistics.

### Text GRU

Whisper `tiny.en` generates transcripts. The transcripts are tokenized, padded, embedded, and passed through a bidirectional GRU.

### DistilBERT Bonus

The vanilla RNN text branch is upgraded with `distilbert-base-uncased` from Hugging Face. This satisfies the transformer bonus challenge, but it does not greatly improve text performance because the transcripts have little semantic emotion content.

### Audio Augmentation Bonus

The training audio set is augmented with background noise while validation and test clips remain clean. This improves generalization and produced the largest performance gain.

The augmentation script also supports optional pitch shifting with `--include-pitch`, but the completed bonus result used background-noise augmentation.

### Late Fusion

Late fusion combines softmax probability outputs from separately trained models. Tested strategies include:

- Average probabilities
- Weighted average probabilities
- Maximum confidence rule

Best result: averaging noise-augmented audio CNN probabilities with DistilBERT probabilities.

## How To Run

Create and activate a Python environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create the dataset index:

```powershell
python src/data/create_index.py
```

Build audio features:

```powershell
python src/features/build_audio_features.py
```

Train the baseline audio CNN:

```powershell
python src/training/train_audio_cnn.py --run-name audio_cnn
```

Generate transcripts:

```powershell
python src/features/generate_transcripts.py
```

Build text features and train the GRU:

```powershell
python src/features/build_text_features.py
python src/training/train_text_rnn.py
```

Run late fusion:

```powershell
python src/fusion/late_fusion.py
python src/evaluation/build_results_table.py
```

Run the bonus augmentation experiment:

```powershell
python src/features/build_augmented_audio_features.py --output-dir data/processed/audio_noise_augmented
python src/training/train_audio_cnn.py --data-dir data/processed/audio_noise_augmented --run-name audio_cnn_noise_augmented
```

Run the DistilBERT bonus branch:

```powershell
python src/training/train_distilbert_text.py
```

## Report

See the full technical report here:

[reports/technical_report.md](reports/technical_report.md)

It includes architecture diagrams, training curves, confusion matrices, dataset challenges, and result analysis.
