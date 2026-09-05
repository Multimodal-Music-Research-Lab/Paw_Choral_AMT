# Third-party provenance

## High-resolution Piano Transcription with Pedals

Several acoustic-model, feature-extraction, post-processing, and utility
components were adapted from ByteDance's
[piano_transcription](https://github.com/bytedance/piano_transcription)
repository, audited at commit
`1ade7dcd4348add669a67c6e6282456c8c6633bd`.

The upstream README identifies the project license as Apache License 2.0. This
repository retains that attribution and distributes its source under the same
license. The original paper is:

Modified descendants in this release are `src/models.py`,
`src/feature_extractor.py`, `src/piano_vad.py`, `src/utilities.py`,
`src/data_generator.py`, `src/losses.py`, `src/main_iter.py`,
`src/evaluate.py`, `src/inference.py`, and `src/calculate_scores.py`. Each
carries an SPDX identifier and a prominent modification notice.

> Q. Kong, B. Li, X. Song, Y. Wan, and Y. Wang, “High-resolution piano
> transcription with pedals by regressing onset and offset times,” IEEE/ACM
> Transactions on Audio, Speech, and Language Processing, 2021.

## Data and model artifacts

No third-party audio, YouChorale annotations, model checkpoints, SoundFonts,
or generated MIDI files are distributed in this snapshot. Those artifacts
have independent terms and must be reviewed file by file before release.
