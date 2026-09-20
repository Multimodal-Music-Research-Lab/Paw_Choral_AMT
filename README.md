# PawCT: Part-Aware Choral Transcription

PawCT transcribes a mixed choral recording into separate note-level soprano,
alto, tenor, and bass (SATB) parts. The repository also provides **PagCT**, its
part-agnostic counterpart, and **Post-VA**, a two-stage symbolic voice-assignment
baseline.

[Online demo](https://hanyu-meng.github.io/Paw_Choral_AMT_Demo/) ·
[Code walkthrough](docs/CODE_WALKTHROUGH.md) ·
[Data format](docs/DATA.md) ·
[Experiment protocol](docs/EXPERIMENT_PLAN.md) ·
[Reproducibility status](docs/REPRODUCIBILITY.md)

> **Release status.** This is an audited research snapshot. The model,
> evaluation, visualization, and release-checking code is included, but the
> exact historical checkpoints and resolved configurations for every manuscript
> row are not yet public. Values listed below are manuscript-reported results,
> not measurements regenerated from this commit.

## Method at a glance

| System | Input | Output |
| --- | --- | --- |
| **PagCT** | mixed choral audio | one merged note track |
| **PawCT** | mixed choral audio | separate SATB note tracks |
| **PawCT + RP** | mixed choral audio | PawCT with a pitch-range prior |
| **PawCT + RP + OC** | mixed choral audio | PawCT with range and ordered-continuity priors |
| **PagCT + Post-VA** | PagCT notes | SATB labels assigned in a second stage |

PawCT uses a shared convolutional recurrent encoder with SATB-specific onset,
offset, and frame heads, plus an auxiliary part-presence head. Union-level
supervision preserves the note content of the mixture. In the current code,
trusted part labels remain fixed: RP suppresses unsupported out-of-range output
mass, while OC regularizes melodic motion between unambiguous consecutive note
events. Neither prior adds an inference-time module.

Main implementations:

- PagCT and PawCT: [`src/models.py`](src/models.py)
- Training losses, including RP and OC: [`src/losses.py`](src/losses.py)
- Target construction: [`src/choral_targets.py`](src/choral_targets.py)
- Post-VA: [`src/train_midi_voice_assignment.py`](src/train_midi_voice_assignment.py)

![Ground truth, PawCT, PagCT, and PagCT plus Post-VA piano rolls](docs/assets/exsultate-deo-four-panel.png)

## Manuscript-reported results

| Task | System | Note F1 @ 50 ms |
| --- | --- | ---: |
| Part-agnostic | Yu et al. (2024) | 0.237 |
| Part-agnostic | **PagCT** | **0.382** |
| Part-aware, macro SATB | PagCT + Post-VA | 0.175 |
| Part-aware, macro SATB | **PawCT + RP + OC** | **0.225** |

These are historical manuscript values. The audit identified differences
between the historical and current target/evaluation protocols, so they must
not be presented as results reproduced by the current commit. Machine-readable
transcriptions are stored in [`repro/expected`](repro/expected); the exact
boundary between reported and reproducible claims is documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Repository structure

```text
src/          models, training, inference, decoding, and evaluation
tools/        audits and manuscript-style visualizations
tests/        unit and release tests
repro/        reported values, frozen splits, and audit records
docs/         technical documentation and the GitHub Pages demo
experiments/  exploratory models not reported in the manuscript
checkpoints/  artifact manifest; model weights are not committed
```

## Setup

Python 3.11 was used in the audited environment. Install the PyTorch build
appropriate for your platform first, then install the remaining dependencies:

```bash
conda create -n pawct python=3.11 -y
conda activate pawct

# Select the appropriate PyTorch command from https://pytorch.org/get-started/locally/
pip install torch==2.10.0 torchaudio==2.10.0
pip install -r requirements-dev.txt

export PYTHONPATH="$PWD/src:$PWD/tools:$PWD/experiments${PYTHONPATH:+:$PYTHONPATH}"
```

The audited lab environment used PyTorch 2.10.0 with CUDA 12.8. CPU execution
is sufficient for tests, but not representative of training speed.

## Data preparation

YouChorale data is not redistributed. Set a local path to a legally obtained
copy with the following structure:

```text
YouChorale/
├── train.json
├── valid.json
├── test.json
├── audio/<recording-id>.(wav|mp3|flac|m4a)
├── midi/<recording-id>.(mid|midi)
└── note/<recording-id>.pkl
```

```bash
export YOUCHORALE_DIR=/absolute/path/to/YouChorale

python src/data_generator.py pack_youchorale_dataset_to_hdf5 \
  dataset.youchorale_dir="$YOUCHORALE_DIR" \
  exp.workspace=./workspaces \
  feature.sample_rate=16000
```

See [`docs/DATA.md`](docs/DATA.md) for annotation semantics, audio-boundary
handling, split requirements, and redistribution restrictions.

## Training

The following is an executable PawCT example, not an exact reproduction of a
manuscript row:

```bash
python src/main_iter.py \
  dataset.train_set=youchorale \
  dataset.test_set=youchorale \
  dataset.youchorale_dir="$YOUCHORALE_DIR" \
  model.arch=pawct \
  model.mode=frame_onset_offset \
  choral.enable=true \
  choral.target_assignment=part_name \
  choral.apply_presence_gate=false \
  feature.max_note_shift=0 \
  exp.workspace=./workspaces \
  exp.random_seed=86 \
  exp.batch_size=8
```

Use `model.arch=pagct` with `choral.enable=false` for part-agnostic training.
RP and OC are enabled through `choral.range_prior_loss_weight` and
`choral.continuity_prior_loss_weight`, respectively. Hyperparameters must be
selected on validation only.

The complete, auditable workflow—including frozen split manifests, RP/OC
settings, inference, validation-only threshold selection, and locked test
evaluation—is specified in
[`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md). Do not tune decoding
thresholds on the test set.

## Main entry points

| Task | Command |
| --- | --- |
| Train PagCT or PawCT | `python src/main_iter.py ...` |
| Run inference | `python src/inference.py ...` |
| Select validation thresholds | `python src/search_best_thresholds.py ...` |
| Score part-aware outputs | `python src/calculate_choral_scores.py ...` |
| Train Post-VA | `python src/train_midi_voice_assignment.py ...` |
| Audit targets | `python tools/audit_choral_targets.py ...` |

Each formal run should record its source commit, resolved configuration,
checkpoint hash, split manifest, probability provenance, and selected
thresholds. See [`repro/README.md`](repro/README.md) for the artifact contract.

## Tests

```bash
python -m compileall -q src tools experiments tests
pytest -q
python scripts/check_release.py
```

The suite covers model forward/backward passes, RP/OC behavior, target and
split integrity, checkpoint compatibility, note decoding, validation/test
separation, probability provenance, and visualization utilities.

## Demo

The dependency-free project page is stored in `docs/`. Preview it locally with:

```bash
python -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>.

## Current limitations

- Exact historical checkpoints and resolved per-row configurations are not yet
  included.
- YouChorale audio, annotations, MIDI files, and generated predictions are not
  redistributed.
- The stable public transcriber currently exposes the merged output; a polished
  audio-to-four-track SATB MIDI command remains release work.
- Post-VA remains exploratory until its training and predicted-note feature
  paths are fully aligned.

Further details are tracked in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## License and attribution

The source code is released under the Apache License 2.0. Portions are adapted
from ByteDance's
[High-resolution Piano Transcription](https://github.com/bytedance/piano_transcription);
see [`NOTICE`](NOTICE) and [`THIRD_PARTY.md`](THIRD_PARTY.md).

The source-code license does not grant rights to datasets, recordings,
annotations, checkpoints, or generated media.

Maintainer: [Hanyu Meng](https://github.com/Hanyu-Meng), Multimodal Music
Research Lab.
