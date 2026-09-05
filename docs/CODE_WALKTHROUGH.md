# Code walkthrough

This document follows one sample from disk to an evaluated SATB transcription.
It describes the code that is present, not an idealized reimplementation.

## 1. Configuration and task resolution

`src/config.yaml` contains Hydra configuration for the dataset, features,
model, loss, decoding, and workspace. `src/utilities.py::get_task_spec`
normalizes legacy model options into a task description used by the trainer,
model factory, loss, and evaluator.

The public defaults are deliberately conservative: data paths are local,
online choral pitch shifting is disabled, and probability generation targets
the validation split. Paper experiments require explicit overrides.

## 2. Data packing and sampling

`src/data_generator.py` converts supported datasets to one HDF5 file per
recording. Each file stores the waveform, MIDI events/times, split, and basic
metadata. `Sampler` creates overlapping 10-second examples at a configurable
hop size; `BasePianoDataset` turns a sampled segment into audio plus merged
frame/onset/offset targets.

For PawCT, `ChoralSATBDataset` additionally reads a per-recording `note/*.pkl`
annotation. Part labels such as S1/S2 or A1/A2 are collapsed to S/A/T/B before
constructing arrays with shape `[time, 4, 88]`.

Three target-assignment modes are central to the manuscript:

- `part_name`: preserve the annotation's named SATB part.
- `range_prior`: assign each event using typical SATB ranges plus a mismatch
  cost.
- `ordered_continuity`: solve each onset group in descending pitch order and
  add melody-continuity and overlap costs; groups larger than four fall back
  to independent assignment.

Missing SATB labels now raise an error. Online note shifting is also blocked
for choral training because the audited implementation shifts audio/global
targets but does not shift the voice-specific targets.

## 3. Acoustic features

`src/feature_extractor.py` implements the time-frequency front end. The paper
configuration uses 16 kHz audio, a 2048-point FFT, a 160-sample hop (100
frames/s), and 229 log-Mel bins.

## 4. PagCT

`src/models.py::FlexibleHPT` is the part-agnostic system. It predicts merged
onset, offset, and frame probabilities and refines the frame stream using event
cues. Its decoded output is a single note-event stream without SATB labels.

Important implementation detail: this class instantiates separate complete
acoustic branches for frame, onset, and offset. PawCT uses one shared encoder,
so these two classes are not parameter-matched merely by changing their heads.

## 5. PawCT

`src/models.py::FlexibleHPTChoralStream` contains the part-aware model:

1. A shared CRNN maps the mixture to a 512-dimensional sequence.
2. Four onset, frame-seed, and offset heads predict S/A/T/B activity.
3. A BiGRU refines the voice-wise frame predictions using detached onset and
   offset cues.
4. A segment-level presence head predicts active parts and softly gates each
   voice output.
5. A max over voices forms merged `frame_output`, `onset_output`, and
   `offset_output` tensors.

The output dictionaries therefore contain both voice tensors
`[batch, time, 4, pitch]` and merged tensors `[batch, time, pitch]`.

## 6. Training objective

`src/losses.py::choral_task_bce` combines masked binary cross-entropy terms for
voice frame/onset/offset predictions, their merged union, and part presence.
The union terms encourage the four heads, collectively, to preserve global
note content.

The code also retains optional assignment regularizers for experimental VA2
modules. Those modules are not part of the reported PawCT-OC system.

The target processor creates precise onset/offset regression rolls, but the
located choral loss supervises binary onset/offset rolls. The regression-style
post-processor still estimates sub-frame shifts from peaks. This mismatch must
be reconciled before claiming that the released training code directly
supervises onset/offset regression.

## 7. Training loop

`src/main_iter.py` builds the selected model/datasets, runs Adam or AdamW,
periodically evaluates segment metrics, and saves checkpoints. It logs to
TensorBoard and optionally Weights & Biases.

The current loop runs for `exp.total_iteration` and saves periodically. It does
not yet implement the manuscript's stated early stopping by validation loss or
best-validation-loss checkpoint selection.

## 8. Decoding and inference

`src/inference.py::PianoTranscriber` frames a complete waveform, performs
overlap inference, stitches the segments, and calls the selected post-
processor in `src/utilities.py`. The post-processors find onset/offset peaks,
pair them with frame activity, and return note events.

Dataset inference writes probability/target bundles to split-isolated paths:

```text
workspaces/probs/<dataset>/<validation-or-test>/<model>/<checkpoint>/
```

`PianoTranscriber.transcribe()` currently decodes the merged union channel.
The paper visualization code contains voice-wise decoding, but a supported
audio-to-four-track SATB MIDI command still needs to promote that logic into
the public inference API.

## 9. Post-VA

`src/train_midi_voice_assignment.py::SymbolicVoiceAssignmentNet` is a
note-sequence classifier. It embeds pitch and pitch class, transforms 11
continuous/numeric features, projects the concatenated representation, then
uses a two-layer bidirectional LSTM to classify every note as S/A/T/B. A second
head predicts which parts are present in the segment.

`src/infer_midi_voice_assignment.py` loads the model, predicts labels over
overlapping note windows, and can write a four-track MIDI. The audio-conditioned
and predicted-MIDI evaluation scripts connect PagCT outputs to Post-VA.

The historical implementation has two scientific caveats: its real training
run used additional range/crossing regularizers, and beat-related features
available for ground-truth notes are zero-filled in some predicted-note paths.

## 10. Evaluation

`src/calculate_scores.py` measures merged frame and note transcription.
`src/calculate_choral_scores.py` decodes and scores S/A/T/B independently,
including 50 ms and 100 ms onset metrics, onset+offset metrics, frame metrics,
and part presence.

`src/search_best_thresholds.py` performs a threshold grid search. The release
version defaults to validation, stores validation and test probabilities
separately, and refuses accidental test-set tuning.

## 11. Visualization and experimental code

`tools/visualize_choral_four_panel_single_song.py` generated the same four-way
layout used in the manuscript: ground truth, PawCT, PagCT, and PagCT + Post-VA.
Other files in `tools/` support batch rendering and legends.

`experiments/` contains symbolic SATB editors and VA2 variants that continued
after the core paper pipeline. They are retained for provenance but should not
be described as paper contributions or stable APIs.
