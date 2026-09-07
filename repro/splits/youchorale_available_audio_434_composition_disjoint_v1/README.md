# YouChorale `available-audio-434` composition-disjoint split v1

This directory freezes a deterministic, composition-disjoint split of the 434
recordings that are both named by the official YouChorale manifests and present
as exact filename stems in the audited historical HDF5 pack. It is a **data
protocol**, not a transcription result.

## Frozen protocol

- Source commit: `a6fa99bc9107d605f4e53057aed0a3ca896596fd`
- Transfer bundle SHA-256:
  `945005bda68cf8bfdf42d2090190065eeb34d5f7183d8698f902eb977a79e712`
- Selected recording-ID-set SHA-256:
  `bedaf5f3a1b8d007217d7c448d4ed34bd968baf4eafc813864fa675b2dea333c`
- Canonical three-manifest content SHA-256 used in checkpoint provenance:
  `c2127b93e0fbd222830700d6115d945cfe90ed67a941b8473816665b60cd45ff`
- Grouping key: normalized `(composer, title)` from public `info.csv`
- Normalization: Unicode NFKC, case-folding, every Unicode punctuation
  character replaced by a space, then whitespace collapsed
- Unicode database version: `14.0.0`
- Ranking: ascending `SHA256(UTF-8("pawct-icassp2027-v1" + group_key))`
- Cuts: floor 80%/90% of ranked groups

Sequence hashes sort exact values lexicographically, encode each as UTF-8 with
a trailing newline, concatenate them, and apply SHA-256. HDF5 matching is
recursive and uses a case-insensitive `.h5`/`.hdf5` extension but an exact,
case-sensitive filename stem. HDF5 contents are not validated by this split
builder.

## Result

| Split | Recordings | Composition groups | Recording-ID-set SHA-256 |
| --- | ---: | ---: | --- |
| Train | 355 | 193 | `aa2ed644a7b09fac8f7afd7fab126aae7e900ba5fc3bfb1bb6ad96ecbc4232ef` |
| Validation | 40 | 24 | `d622d482ed8faa2462a2b983e7fea3879935cfcbfb2d549c6a67315be4089cf7` |
| Test | 39 | 25 | `0e774c0fe388459e73af4fb7060a19d00aac0d836c828a300ae065ae01bd9a10` |

Train/validation, train/test, and validation/test each have exactly zero shared
normalized composition groups. The official recording split is retained only
as a separate prior-work comparison; it must not be mixed with this protocol.
Every model row compared on this protocol must be retrained from scratch using
the same manifests, seeds, selection rule, and validation-only thresholds.

## Files

- `group_map.json`: normalized groups, deterministic ranks, assignments, and
  recording-to-group mapping.
- `train.json`, `valid.json`, `test.json`: frozen recording manifests.
- `audit_report.json`: source coverage, official-split leakage audit, generated
  split counts, hashes, and zero-overlap verification.
- `validation.json`: compact independent assertions from the remote execution.

## Reproduction

From the repository root, using a trusted local YouChorale copy and the exact
audited HDF5 pack:

```bash
python tools/build_youchorale_composition_split.py \
  --dataset-dir "$YOUCHORALE_DATASET_DIR" \
  --packed-hdf5-dir "$YOUCHORALE_HDF5_DIR" \
  --output-dir repro/splits/youchorale_available_audio_434_composition_disjoint_v1 \
  > /tmp/youchorale-composition-split-audit.json
cmp /tmp/youchorale-composition-split-audit.json \
  repro/splits/youchorale_available_audio_434_composition_disjoint_v1/audit_report.json
```

An identical rerun is a no-op. A different or partial frozen output fails
closed. The final CPU-only remote run invoked the generator twice and produced
byte-identical reports. A prior isolated run at `8ccdc41` was not published
because its external validation script compared two different declared hash
encodings; `a6fa99b` aligned the generator with the repository-wide convention
before this split was frozen.

## File integrity

```text
91476eec51df46b31a03bd0330d751a7eaef4d7224ca0bb5738b5d82bb247d05  audit_report.json
d447c0d4bc7a9d722a1e67a5f67e45f622a3b1c995f866fbb957464423882d7c  group_map.json
5c44353da8ea5aa4daf0708a73cdecd5d85ab0e9b343baef8e73b2facf91406e  test.json
01701c53dc32a1a6f355126b0918f676dd0a9206c321b1efd1128b797a1ce2ff  train.json
32913ce955b3826c85372cffd31f1be8d94aaf0877d5a53a66d6034ca7e9bbfb  valid.json
2ead4e11668566983497f75a1f4e3a9dccbaf2af9ddb18799b3adc3738297b67  validation.json
```
