# YouChorale runnable-pack target audit at `4f23e77`

This is a **split-coverage and target-semantics audit**, not a transcription
performance result. It restricts each source manifest to recording stems that
exist in the historical `youchorale_sr16000` HDF5 directory, then computes the
same descriptive label/RP/OC statistics over exactly that runnable subset.

## Provenance and scope

- Source commit: `4f23e775ee2a18c0eed16505756973081d20d3ee`
- Transfer bundle SHA-256:
  `087cbd6747c124688cc2721a13f52f592b7069a82725a34fde455f9bfd761fa3`
- Packed recording-stem set SHA-256:
  `bedaf5f3a1b8d007217d7c448d4ed34bd968baf4eafc813864fa675b2dea333c`
- Execution: CPU-only, one OpenBLAS/OpenMP thread, low-priority process on the
  `lab5090` worker. No GPU process or persistent session was changed.
- Dataset and historical HDF5 roots were read only; no audio, MIDI, notes,
  HDF5, or predictions are redistributed.

Coverage uses recursive, exact, case-sensitive filename-stem existence. It
does not hash or validate HDF5 contents. The full ID sets and missing packed
stems are recorded in each JSON report. A separate source-file check confirmed
that the 18 manifest items without HDF5 also lack a supported audio file in the
trusted local dataset copy, although their MIDI and note annotations exist.

## Manifest coverage and target results

| Split | Manifest recordings | Packed intersection | Coverage | Audited notes | Canonical known | >4-note onset groups | Duplicate canonical-voice groups | Modern RP/OC changes | Legacy RP changes | Legacy OC changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 392 | 376 | 95.92% | 285,674 | 100.00% | 14.60% | 33.72% | 0.00% | 0.27% | 14.47% |
| Validation | 30 | 28 | 93.33% | 22,930 | 100.00% | 17.81% | 43.99% | 0.00% | 1.30% | 20.00% |
| Test | 30 | 30 | 100.00% | 28,324 | 100.00% | 16.17% | 38.80% | 0.00% | 1.71% | 19.07% |
| All | 452 | 434 | 96.02% | 336,928 | 100.00% | 14.94% | 34.83% | 0.00% | 0.46% | 15.24% |

The manuscript describes 452 recordings, but this historical acoustic pack can
load only 434. A result produced from it must therefore be labelled as the
`available-audio-434` protocol, with the split hashes above, rather than as a
complete 452-recording run. The alternatives are to recover the 18 missing
audio files legally and repack them, or to freeze and publish this reduced
protocol. Missing files must not redefine the split silently.

## RP range for the runnable training subset

| Voice | Train p01 | Train p99 | Observed min | Median | Observed max |
| --- | ---: | ---: | ---: | ---: | ---: |
| S | 60 | 79 | 55 | 70 | 83 |
| A | 55 | 74 | 42 | 65 | 78 |
| T | 50 | 70 | 36 | 59 | 79 |
| B | 41 | 62 | 34 | 52 | 69 |

These p01/p99 values, plus the fixed two-semitone margin, are the primary RP
pilot setting for the current runnable pack:

```text
mins=[60,55,50,41]
maxs=[79,74,70,62]
margin=2.0
```

The tenor p99 is 70 for the actual 376-recording training subset, not 69 from
the annotation-only 392-recording manifest audit. The runnable-pack statistic
therefore supersedes the earlier value for experiments using this HDF5 pack.
The loss still excludes trusted annotated positives, so genuine out-of-range
training notes are never penalized.

## Reproduction command

```bash
export YOUCHORALE_DIR=/absolute/path/to/YouChorale
export YOUCHORALE_HDF5_DIR=/absolute/path/to/youchorale_sr16000
for split in train validation test all; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python tools/audit_choral_targets.py \
      --dataset-dir "$YOUCHORALE_DIR" \
      --packed-hdf5-dir "$YOUCHORALE_HDF5_DIR" \
      --split "$split" \
      --output-json "${split}.json"
done
```

## File integrity

```text
418646dca8360c83d947de450500ed1f143871ddb4e1e55a4ecb23af93cd8720  all.json
85044d996ae5900846002b5d6002d75d7c579fe039ca9e418d578592a2c553af  test.json
c42fc36f42554ba255fc953131d51baef6b9c079bf8331f2d389d210dfbc26dc  train.json
4a291dbd378a1aee56e8b9742f66f830c85dc857c4ad08d43514e5c0f8338852  validation.json
```
