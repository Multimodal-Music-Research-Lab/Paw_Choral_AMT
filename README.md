# PawCT: Part-Aware Choral Transcription

Research code for **“Toward Part-Aware Choral Transcription with Singing
Voice Assignment.”** PawCT transcribes a mixed choral recording into
note-level soprano, alto, tenor, and bass (SATB) parts. This repository also
contains the part-agnostic transcriber (PagCT) and a two-stage symbolic voice
assignment baseline (Post-VA).

> **Release status — audited research snapshot.** The core methods and the
> script used for the manuscript's qualitative figure are present. The exact
> training manifests, frozen configurations, checkpoints, and a one-command
> Table 1/2 reproduction are not yet available. Paper numbers below are
> reported results, not results regenerated from this commit. See
> [Reproducibility status](docs/REPRODUCIBILITY.md) before citing numerical
> claims.

[Static demo](docs/index.html) ·
[Code walkthrough](docs/CODE_WALKTHROUGH.md) ·
[Data contract](docs/DATA.md) ·
[Reproducibility audit](docs/REPRODUCIBILITY.md)

## What is implemented

| System | Input → output | Main implementation |
| --- | --- | --- |
| **PagCT** | mixed audio → merged note events | `src/models.py::FlexibleHPT` |
| **PawCT** | mixed audio → SATB note events | `src/models.py::FlexibleHPTChoralStream` |
| **PawCT-RP / PawCT-OC** | PawCT with range-prior or ordered-continuity targets | `src/data_generator.py::ChoralSATBDataset` |
| **Post-VA** | merged note sequence → SATB labels | `src/train_midi_voice_assignment.py::SymbolicVoiceAssignmentNet` |

PawCT uses one shared CRNN encoder, four voice-specific onset/frame/offset
heads, a segment-level part-presence gate, and a max-over-parts union output.
The training objective combines per-part losses with union and presence losses.
RP and OC change how ambiguous source annotations are assigned to SATB targets;
they are not extra inference modules.

![Ground truth, PawCT, PagCT, and PagCT plus Post-VA piano rolls](docs/assets/exsultate-deo-four-panel.png)

The figure is a precomputed author-generated visualization for the manuscript
example “Exsultate Deo” (`jd2_r4PK5dc`). It is included to verify code/figure
provenance; the underlying YouChorale audio and annotations are not
redistributed.

## Paper-reported results

The manuscript reports the following headline numbers on its YouChorale test
split:

| System | Part-agnostic note F1 @ 50 ms |
| --- | ---: |
| Yu et al. (2024) | 0.237 |
| PagCT without transposition augmentation | 0.333 |
| **PagCT** | **0.382** |

| System | Average SATB frame F1 | Average SATB note F1 @ 50 ms | Note retention ratio |
| --- | ---: | ---: | ---: |
| PawCT without union loss | 0.379 | 0.190 | 49.74% |
| PawCT | 0.458 | 0.217 | 56.81% |
| PawCT-RP | 0.510 | 0.209 | 54.71% |
| **PawCT-OC** | 0.503 | **0.225** | **58.90%** |
| PagCT + Post-VA | 0.489 | 0.175 | 45.81% |

We call the manuscript's “VA Rate” a **note retention ratio** here because its
denominator comes from a separate part-agnostic system; it should not be read
as a pure voice-classification accuracy. Machine-readable transcriptions of
the reported tables are in [`repro/expected`](repro/expected).

## Repository layout

```text
src/          core acoustic, Post-VA, inference, and evaluation code
tools/        paper-style visualization utilities
experiments/  symbolic editor and VA2 prototypes not reported in the paper
tests/        existing unit tests plus release checks
repro/        reported values and the reproduction contract
docs/         static project/demo page and technical documentation
checkpoints/  artifact manifest only; model files are not committed
```

The modules intentionally remain flat to preserve the audited research code's
import behavior. Run commands from the repository root with `src`, `tools`, and
`experiments` on `PYTHONPATH`.

## Installation

Python 3.11 was used in the audited lab environment. Create an isolated
environment, install the PyTorch build appropriate for your CUDA runtime, and
then install the remaining pinned dependencies:

```bash
conda create -n pawct python=3.11 -y
conda activate pawct

# Choose the correct command for your platform at pytorch.org first.
pip install torch==2.10.0 torchaudio==2.10.0
pip install -r requirements-dev.txt

export PYTHONPATH="$PWD/src:$PWD/tools:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
```

The original lab environment used PyTorch 2.10.0 with CUDA 12.8 on an RTX
5090. A CPU install is sufficient for syntax/unit checks but not representative
of training throughput.

## Data

Data are intentionally excluded. Point Hydra at a local, legally obtained
dataset directory:

```bash
export YOUCHORALE_DIR=/absolute/path/to/YouChorale
```

The expected YouChorale layout is:

```text
$YOUCHORALE_DIR/
├── train.json
├── valid.json
├── test.json
├── audio/
│   └── <recording-id>.(wav|mp3|flac|m4a)
├── midi/
│   └── <recording-id>.(mid|midi)
└── note/
    └── <recording-id>.pkl
```

The packer uses the literal `audio/` and `midi/` directory names and pairs
files by recording ID. SATB label pickles are required for PawCT: missing
labels now fail fast instead of being silently treated as empty targets. See
[docs/DATA.md](docs/DATA.md) for the annotation schema, split requirements,
and redistribution policy.

## Prepare HDF5 features

Packing writes to `exp.workspace/hdf5s/...`; this generated directory is
ignored by Git.

```bash
python src/data_generator.py pack_youchorale_dataset_to_hdf5 \
  dataset.youchorale_dir="$YOUCHORALE_DIR" \
  exp.workspace=./workspaces \
  feature.sample_rate=16000
```

Before training, verify that train/validation/test are composition-disjoint,
not merely recording-disjoint. The current repository does not yet ship the
manuscript's split manifest.

## Train

### PagCT

```bash
python src/main_iter.py \
  dataset.train_set=youchorale \
  dataset.test_set=youchorale \
  dataset.youchorale_dir="$YOUCHORALE_DIR" \
  model.arch=hpt \
  model.mode=frame_onset_offset \
  choral.enable=false \
  exp.workspace=./workspaces \
  exp.total_iteration=200000 \
  exp.batch_size=8
```

### PawCT-OC

```bash
python src/main_iter.py \
  dataset.train_set=youchorale \
  dataset.test_set=youchorale \
  dataset.youchorale_dir="$YOUCHORALE_DIR" \
  model.arch=hpt \
  model.mode=frame_onset_offset \
  choral.enable=true \
  choral.voice_assignment_method=ordered_continuity \
  choral.union_frame_loss_weight=1.0 \
  choral.union_onset_loss_weight=1.0 \
  choral.union_offset_loss_weight=1.0 \
  feature.max_note_shift=0 \
  exp.workspace=./workspaces \
  exp.total_iteration=200000 \
  exp.batch_size=8
```

`feature.max_note_shift` must remain zero for `ChoralSATBDataset` in this
snapshot: the older online path shifted mixture/global targets but not SATB
labels. Use only a synchronously generated offline transposition dataset until
that path has an end-to-end test.

These commands express the manuscript's intended setup; they are **not yet
frozen reproduction commands**. In particular, the located historical runs
continued to 300k steps and the current trainer does not implement the
manuscript's stated validation-loss early stopping.

## Validation thresholds, then locked test

Inference and scoring now keep validation and test probabilities in separate
directories. Threshold search defaults to validation and refuses test tuning
unless a user explicitly marks it as a non-reportable diagnostic.

```bash
# 1. Produce validation probabilities.
python src/inference.py \
  dataset.test_set=youchorale \
  dataset.eval_split=validation \
  dataset.youchorale_dir="$YOUCHORALE_DIR" \
  choral.enable=true \
  choral.voice_assignment_method=ordered_continuity \
  post.post_processor_type=regression \
  exp.workspace=./workspaces \
  exp.ckpt_iteration=<ITERATION>

# 2. Select thresholds on validation only.
python src/search_best_thresholds.py \
  --config_dir src \
  --workspace ./workspaces \
  --test_set youchorale \
  --split validation \
  --ckpt_iteration <ITERATION> \
  --choral_enable \
  --choral_per_voice \
  --voice-assignment-method ordered_continuity \
  --post_processor_type regression \
  --output_txt ./workspaces/thresholds/pawct_oc_validation.txt

# 3. Copy the selected thresholds into a frozen config, produce test
#    probabilities once, and evaluate without another search.
python src/inference.py \
  dataset.test_set=youchorale \
  dataset.eval_split=test \
  dataset.youchorale_dir="$YOUCHORALE_DIR" \
  choral.enable=true \
  choral.voice_assignment_method=ordered_continuity \
  post.post_processor_type=regression \
  exp.workspace=./workspaces \
  exp.ckpt_iteration=<ITERATION> \
  choral.use_per_voice_thresholds=true \
  choral.voice_frame_thresholds='[...]' \
  choral.voice_onset_thresholds='[...]' \
  choral.voice_offset_thresholds='[...]'

python src/calculate_choral_scores.py \
  dataset.test_set=youchorale \
  dataset.eval_split=test \
  dataset.youchorale_dir="$YOUCHORALE_DIR" \
  choral.enable=true \
  choral.voice_assignment_method=ordered_continuity \
  post.post_processor_type=regression \
  exp.workspace=./workspaces \
  exp.ckpt_iteration=<ITERATION> \
  choral.use_per_voice_thresholds=true \
  choral.voice_frame_thresholds='[...]' \
  choral.voice_onset_thresholds='[...]' \
  choral.voice_offset_thresholds='[...]'
```

## Post-VA

Train the symbolic note classifier from ground-truth note sequences:

```bash
python src/train_midi_voice_assignment.py \
  --dataset-dir "$YOUCHORALE_DIR" \
  --workspace ./workspaces \
  --experiment-name post_va \
  --learning-rate 1e-4
```

The historical checkpoint located during audit was actually trained with
AdamW at `1e-3`, for 80 epochs, with range and crossing regularizers. Its paper
configuration must therefore be reconciled before reporting a reproduced
Post-VA number. Predicted-MIDI evaluation no longer depends on an adjacent
YourMT3 checkout; it uses the declared `mido` dependency.

## Tests

```bash
python -m compileall -q src tools experiments tests
pytest -q
python scripts/check_release.py
```

The test suite includes a data-free PawCT forward/union/loss-backward smoke
test, the local MIDI reader, threshold-split guards, target construction,
metric helpers, and visualization. PagCT parity, real-data integration,
data-integrity, and table-regression tests remain release work; they are tracked in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Demo

`docs/` is a dependency-free GitHub Pages site. It presents the method,
reported tables, and an interactive view of the author-generated qualitative
figure. It deliberately does not offer browser inference or redistribute
YouChorale audio/MIDI. A future listening demo should use only recordings and
derived assets with explicit redistribution permission.

To preview locally:

```bash
python -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>.

## Known limitations

- Exact checkpoints, commit/config hashes, split manifests, and table
  reproduction scripts are not yet released.
- PagCT and PawCT are not parameter-matched in this snapshot: the PagCT class
  has separate full frame/onset/offset acoustic branches, whereas PawCT uses a
  shared encoder.
- Regression-style post-processing is present, but the located loss supervises
  binary onset/offset rolls rather than the generated regression targets.
- Post-VA has a feature mismatch between ground-truth-note training and some
  predicted-note evaluation paths, where beat-related features are zero-filled.
- `PianoTranscriber.transcribe()` writes the merged union output; a stable
  public audio-to-four-track SATB MIDI CLI still needs to be extracted from the
  visualization helpers.
- Files in `experiments/` (symbolic editors, VA2 variants, and related ideas)
  are exploratory and were not reported in the manuscript.

## Security and artifact trust

Only load checkpoints and pickle files from sources you trust. Python pickle
and legacy PyTorch checkpoints can execute code while loading. Generated
artifacts are ignored by default; a future checkpoint release must include its
exact config, source commit, dataset/split manifest, and SHA-256 checksum.

## License and attribution

Source code is released under Apache License 2.0. Portions are adapted from
ByteDance's
[High-resolution Piano Transcription](https://github.com/bytedance/piano_transcription)
codebase; see [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md).
This source-code license does not grant rights to datasets, recordings,
annotations, checkpoints, or generated media.
