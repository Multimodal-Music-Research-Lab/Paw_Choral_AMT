# YouChorale composition-disjoint target audit at `bde12f6`

This is a **data and target-semantics audit**, not a transcription-performance
result. It applies the frozen `available-audio-434` composition-disjoint v1
manifests to the exact packed acoustic subset and audits source annotations
without changing them.

## Provenance and scope

- Source commit: `bde12f65008d61b2e72872ea0345732d6ad9c1b5`
- Transfer bundle SHA-256:
  `3a72af72b6165470f836ae47c6aeb09be646c0928e746dd768f8e0658508dd11`
- Split identity:
  `c2127b93e0fbd222830700d6115d945cfe90ed67a941b8473816665b60cd45ff`
- Packed recording-stem set SHA-256:
  `bedaf5f3a1b8d007217d7c448d4ed34bd968baf4eafc813864fa675b2dea333c`
- Execution: CPU-only, one OpenBLAS/OpenMP/MKL thread, low-priority process.
  The pre-existing GPU process and persistent session were unchanged.
- Dataset annotations and packed HDF5 files were read only. No audio, MIDI,
  HDF5, or predictions are redistributed.

Every selected manifest item has an exact, case-sensitive HDF5 filename-stem
match. HDF5 contents are not opened or validated by this audit.

## Target results

| Split | Recordings | Notes | Canonical known | >4-note onset groups | Duplicate canonical-voice groups | Modern RP/OC changes | Legacy RP changes | Legacy OC changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 355 | 282,493 | 100.00% | 14.86% | 34.97% | 0.00% | 0.26% | 15.03% |
| Validation | 40 | 28,002 | 100.00% | 12.51% | 29.79% | 0.00% | 2.24% | 17.25% |
| Test | 39 | 26,433 | 100.00% | 18.53% | 38.97% | 0.00% | 0.66% | 15.35% |

The test subset contains more greater-than-four-note and duplicate-voice onset
groups than the train subset. This motivates a pre-registered difficulty
breakdown; it does not itself demonstrate an OC benefit. Modern RP and OC
retain every trusted label, so their acoustic effect must come from the soft
output losses rather than from hard target reassignment.

## Frozen RP range for composition-disjoint training

| Voice | Train p01 | Train p99 | Observed min | Median | Observed max |
| --- | ---: | ---: | ---: | ---: | ---: |
| S | 60 | 79 | 53 | 70 | 83 |
| A | 55 | 74 | 42 | 65 | 78 |
| T | 50 | 69 | 36 | 59 | 79 |
| B | 41 | 62 | 34 | 53 | 69 |

Composition-disjoint P2/P3 runs must therefore freeze:

```text
mins=[60,55,50,41]
maxs=[79,74,69,62]
margin=2.0
```

The tenor p99 differs from the official runnable split (`69` versus `70`).
Each protocol must use statistics estimated only from its own training
manifest. Trusted positive labels remain excluded from the RP penalty.

## Reproduction

```bash
export YOUCHORALE_DATASET_DIR=/absolute/path/to/YouChorale
export YOUCHORALE_HDF5_DIR=/absolute/path/to/youchorale_sr16000
export SPLIT_DIR=repro/splits/youchorale_available_audio_434_composition_disjoint_v1

python tools/audit_choral_targets.py \
  --dataset-dir "$YOUCHORALE_DATASET_DIR" --packed-hdf5-dir "$YOUCHORALE_HDF5_DIR" \
  --split train --recording-manifest "$SPLIT_DIR/train.json" --output-json train.json
python tools/audit_choral_targets.py \
  --dataset-dir "$YOUCHORALE_DATASET_DIR" --packed-hdf5-dir "$YOUCHORALE_HDF5_DIR" \
  --split validation --recording-manifest "$SPLIT_DIR/valid.json" --output-json validation.json
python tools/audit_choral_targets.py \
  --dataset-dir "$YOUCHORALE_DATASET_DIR" --packed-hdf5-dir "$YOUCHORALE_HDF5_DIR" \
  --split test --recording-manifest "$SPLIT_DIR/test.json" --output-json test.json
```

## File integrity

```text
ca2f546ff3cd7526aa6a2b2cf10def49cb89c2ca67f1af4d15a825ef87ecde53  train.json
684f0b3c62f952439fb80acf9c1a6a6199479f6c6f66ec887c6a84c9d509fc11  validation.json
60136f604111fe7ec4c674c9431a8fcd392858f67245cc0df1b31107d8bacc32  test.json
```
