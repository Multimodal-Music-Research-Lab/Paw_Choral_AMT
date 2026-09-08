# Data contract and redistribution policy

## YouChorale inputs

The acoustic pipeline expects paired files under the literal `audio/` and
`midi/` directory names, plus split metadata. Supported audio extensions are
`.wav`, `.mp3`, `.flac`, and `.m4a`; MIDI extensions are `.mid` and `.midi`.
Files are paired by their recording ID (the filename without its extension).
Both PagCT and PawCT use one trusted pickle annotation per choral recording
under `note/`; HDF5 remains the audio container. Each
pickle contains a list of measure dictionaries; SATB keys such as `S1`, `S2`,
`A1`, `T2`, and `B1` map to note rows whose first, fourth, and fifth fields are
MIDI pitch, onset time, and offset time.

Formal scoring fails closed unless the top-level object is a materialized
`list`, every note has a recognized SATB/divisi label, integer in-range MIDI
pitch, and finite positive duration. Recording-boundary handling is explicit:
the default `strict` policy rejects any out-of-bounds time. The frozen
`clip_offsets_drop_unobservable_onsets_v1` policy clips the offset of a note
whose onset is audible but release is right-censored by the waveform, and
drops a note whose onset itself lies outside the recording. It never changes
the source pickle. Use the same policy for inference, threshold selection, and
scoring, and record it in probability provenance. If two
divisi notes in the same canonical voice share a pitch and quantize to the same
onset frame, the binary voice head cannot represent them separately; both
training targets and references therefore merge them using the earliest onset
and latest offset. Voice targets keep cross-voice unisons distinct. Their
part-agnostic canonical union merges only equal-pitch attacks on the same model
frame, retains later attack frames as rearticulation boundaries, and spans each
overlapping activity component to its true final release. This one projector is
used for PagCT targets, PawCT union targets, inference references, and union
scoring; metric onset tolerance never changes it.

An exhaustive audit of the runnable `available-audio-434` pack found 336,928
source notes: 55 offsets after the packed waveform and 15 attacks after it.
All 15 unobservable attacks and 50 of the 55 right-censored releases are in the
training split. Validation has one affected recording (`yMc5qPZf_gc`) with five
audible final attacks whose releases extend 0.292317225 s beyond the waveform;
test has no boundary violations. Thus the versioned policy changes no test
reference, retains all validation attacks, and makes the already implicit
training boundary behavior explicit.

Because pickle loading can execute code, never use annotations from an
untrusted source. A future dataset release should replace or accompany these
files with a documented JSON/Parquet representation.

## Split requirements

By default the repository reads `train`, `validation`, and `test` attributes
inside packed HDF5 files, corresponding to `train.json`, `valid.json`, and
`test.json` in the source directory. For YouChorale,
`dataset.youchorale_split_dir` selects a frozen external three-manifest
protocol and overrides those historical attributes consistently in training,
inference, threshold search, and scoring.

Before reporting results:

1. Compare each source split manifest with the packed HDF5 stems. Either require
   complete coverage or publish an explicitly named available-audio subset and
   its recording-ID hash; never let missing packs silently redefine a split.
2. Resolve every recording to a composition/work identifier.
3. Verify that no composition, score edition, or derivative recording crosses
   train/validation/test.
4. Check for duplicate and near-duplicate audio.
5. Record the ordered stem list and SHA-256 digest for every split.
6. Use validation for checkpoint/threshold selection and evaluate test once.

The original research snapshot did not include a machine-verifiable
composition-disjoint audit. An audit of the public manifests using exact
normalized `(composer, title)` metadata finds that 22/29 test works overlap
train and 23/30 test recordings belong to a train-seen work. The current split
therefore must not be presented as composition-disjoint.

The located historical HDF5 pack is also incomplete relative to those source
manifests: it contains 376/392 train, 28/30 validation, and 30/30 test stems.
For the resulting `available-audio-434` protocol, 21/29 test works overlap the
runnable train set and 22/30 test recordings belong to a train-seen work. Its
exact stem hashes and missing IDs are archived in the
[runnable-pack audit](../repro/audits/youchorale_packed_targets_20260908_4f23e77/README.md).
The separately frozen
[`available-audio-434` composition-disjoint v1 protocol](../repro/splits/youchorale_available_audio_434_composition_disjoint_v1/README.md)
regroups those same 434 IDs into 355/40/39 recordings with zero normalized-work
overlap for stronger out-of-composition evaluation.

## Pitch transposition

The manuscript describes seven fixed transpositions (`-3` through `+3`
semitones). That offline generation pipeline was not present in the audited
source. Choral dataset loaders therefore reject nonzero online `max_note_shift`:
the older path shifted audio and packed targets but not `note.pkl` targets.

Any replacement augmentation pipeline must shift waveform, merged targets,
voice targets, range annotations, and metadata synchronously, then test the
alignment on known notes.

## Redistribution

This repository does not distribute YouChorale audio, annotations, source
MIDI, predicted MIDI, or model weights. A composition being public domain does
not automatically clear a particular recording, modern edition, annotation,
or derived audio render.

For every future demo asset, add an entry to `docs/assets/manifest.json` with:

- source and stable URL/DOI;
- creator and copyright holder;
- exact license and attribution text;
- excerpt boundaries and transformations;
- hashes of source and derived files;
- whether model predictions may be redistributed.
