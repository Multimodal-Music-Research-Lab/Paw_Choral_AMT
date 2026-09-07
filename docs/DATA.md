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
pitch, finite positive duration, and times inside the packed recording. If two
divisi notes in the same canonical voice share a pitch and quantize to the same
onset frame, the binary voice head cannot represent them separately; both
training targets and references therefore merge them using the earliest onset
and latest offset. Voice targets keep cross-voice unisons distinct. Their
part-agnostic canonical union merges only equal-pitch attacks on the same model
frame, retains later attack frames as rearticulation boundaries, and spans each
overlapping activity component to its true final release. This one projector is
used for PagCT targets, PawCT union targets, inference references, and union
scoring; metric onset tolerance never changes it.

Because pickle loading can execute code, never use annotations from an
untrusted source. A future dataset release should replace or accompany these
files with a documented JSON/Parquet representation.

## Split requirements

The repository expects `train`, `validation`, and `test` labels inside packed
HDF5 files, corresponding to `train.json`, `valid.json`, and `test.json` in the
source directory.

Before reporting results:

1. Resolve every recording to a composition/work identifier.
2. Verify that no composition, score edition, or derivative recording crosses
   train/validation/test.
3. Check for duplicate and near-duplicate audio.
4. Record the ordered stem list and SHA-256 digest for every split.
5. Use validation for checkpoint/threshold selection and evaluate test once.

The original research snapshot did not include a machine-verifiable
composition-disjoint audit. An audit of the public manifests using exact
normalized `(composer, title)` metadata finds that 22/29 test works overlap
train and 23/30 test recordings belong to a train-seen work. The current split
therefore must not be presented as composition-disjoint.

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
