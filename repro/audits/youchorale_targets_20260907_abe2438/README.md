# YouChorale target audit at `abe2438`

This directory records a **data/target-semantics audit**, not model
performance. It contains no YouChorale audio, MIDI, note annotations, recording
IDs, checkpoints, or predictions.

## Provenance

- Source commit: `abe243877b995b528c096c831eb3f884af2fb52d`
- Transfer bundle SHA-256:
  `68e4bc34adda24666657b23e404690f70c5a50e2ad108db5dca578c92cde1dd1`
- Execution: CPU-only, one OpenBLAS/OpenMP thread, low-priority process on the
  `lab5090` worker; the active GPU experiment was left untouched.
- Dataset access: read-only trusted local YouChorale copy. The dataset is not
  redistributed.
- Audit schema: `1`; modeled MIDI range `21..108`; onset grouping rate
  `100 frames/s`.

The split reports can be regenerated from a legally obtained trusted copy:

```bash
export YOUCHORALE_DIR=/absolute/path/to/YouChorale
for split in train validation test all; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python tools/audit_choral_targets.py \
      --dataset-dir "$YOUCHORALE_DIR" \
      --split "$split" \
      --output-json "${split}.json"
done
```

The command loads trusted `note/*.pkl` files, but never modifies the dataset.

## Results

| Split | Recordings | Notes | Canonical SATB known | Onset groups with >4 notes | Groups with duplicate canonical voice | Modern RP changed labels | Modern OC changed labels | Legacy RP changed labels | Legacy OC changed labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 392 | 299,638 | 100.00% | 14.47% | 33.61% | 0.00% | 0.00% | 0.41% | 14.71% |
| Validation | 30 | 25,239 | 100.00% | 19.94% | 46.36% | 0.00% | 0.00% | 1.20% | 19.09% |
| Test | 30 | 28,324 | 100.00% | 16.17% | 38.80% | 0.00% | 0.00% | 1.71% | 19.07% |
| All | 452 | 353,201 | 100.00% | 14.98% | 34.89% | 0.00% | 0.00% | 0.57% | 15.37% |

`Modern RP/OC changed labels` uses identical source-note denominators and
`preserve_known_part_labels=true`. The zero values are expected: every
in-range note in this dataset copy has a canonical S/A/T/B or divisi-derived
label. Consequently, target assignment alone cannot distinguish the modern RP
and OC experiments on this data. Their scientific effect must come from soft
output regularizers that leave annotated positives intact. The legacy OC
transformation instead overwrites many trusted labels and is unsafe as the
main training target.

Validation and test contain more greater-than-four-note and
duplicate-canonical-voice onset groups than train. That shift motivates
difficulty-stratified and composition-level analysis rather than relying only
on a single global mean.

## Annotation-manifest RP range

Across all 392 annotated train-manifest items, the train-only p01/p99 MIDI pitch
ranges are:

| Voice | p01 | p99 | Observed min | Median | Observed max |
| --- | ---: | ---: | ---: | ---: | ---: |
| S | 60 | 79 | 50 | 70 | 83 |
| A | 55 | 74 | 42 | 65 | 79 |
| T | 50 | 69 | 36 | 59 | 79 |
| B | 41 | 62 | 34 | 52 | 69 |

With the configured two-semitone margin, the loss suppresses unsupported
probability mass only outside the robust range plus margin; it never penalizes
a trusted annotated positive. The previous broad ranges
`[60,55,48,40]..[88,79,72,67]` remain a named diagnostic ablation rather than
the primary setting. This annotation-only range must not be applied blindly to
an incomplete acoustic pack: the
[runnable-pack audit](../youchorale_packed_targets_20260908_4f23e77/README.md)
finds that the current 376-recording training pack has tenor p99 = 70 and is the
authoritative range source for experiments using that pack.

## File integrity

```text
f69b70f6c906078fbeaff552e1a4ddd8695ae290b4e338b819a6efa8457e64f2  all.json
f269de17d192cbd42f9a700d9f4443210f5ed8b68e8d83576d4a3d3cc35384c1  test.json
d9f8962562fa24ea8b50b865fb8c53166537054117287afcbdda995dbdf0edf9  train.json
7b1d17fae61cbeb1b016a7e38488bde5e0cc9d12b2a15f9faf851cb42f36241b  validation.json
```
