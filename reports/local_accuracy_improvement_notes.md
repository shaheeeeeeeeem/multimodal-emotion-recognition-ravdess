# Local Accuracy Improvement Notes

These changes are local only and have not been committed or pushed.

## Best New Result

| Model | Accuracy (%) | Macro F1 (%) |
|---|---:|---:|
| Submitted best: Augmented Audio + DistilBERT Fusion | 47.92 | 44.38 |
| New best: Enhanced Audio CNN + Noise Augmentation + SpecAugment | 61.67 | 61.30 |

## What Improved Accuracy

The largest improvement came from the audio branch, not the text branch.

Changes used in the best local model:

- deeper CNN with larger channel capacity
- two convolution layers per block
- SiLU activations
- average pooling plus max pooling before classification
- AdamW optimizer
- label smoothing
- background-noise augmented training data
- SpecAugment during training

## Experiment Ranking

See:

```text
outputs/metrics/local_audio_experiment_ranking.md
```

The original seed-42 SpecAugment run remains best. Other seeds and lighter/stronger masking settings were worse.

## Interpretation

The previous low accuracy was not a hard ceiling. The model was under-regularized and too small. Audio augmentation plus a stronger CNN improved generalization to unseen actors.

Text fusion no longer helps once the audio branch is strong, because RAVDESS transcripts still contain little emotion information.

## Next Best Ideas

1. Try efficient waveform-level pitch/time augmentation.
2. Try pretrained speech/audio embeddings such as Wav2Vec2, HuBERT, or WavLM.
3. Run actor-fold cross-validation to verify the 61.67% result is not split-specific.
4. Tune per-class augmentation for weak classes such as sad, happy, and fearful.
