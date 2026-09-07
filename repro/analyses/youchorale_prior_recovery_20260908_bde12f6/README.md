# RP/OC training-label recovery diagnostic at `bde12f6`

This is a deterministic **training-label mechanism diagnostic**, not an
acoustic transcription result. It masks nested 10%, 25%, and 50% subsets of
canonical training-note voice labels, estimates ranges only from the remaining
visible labels, and asks the existing RP and OC assignment mechanisms to
recover the masked labels.

## Provenance and safeguards

- Source commit: `bde12f65008d61b2e72872ea0345732d6ad9c1b5`
- Transfer bundle SHA-256:
  `3a72af72b6165470f836ae47c6aeb09be646c0928e746dd768f8e0658508dd11`
- Protocol: `pawct_train_label_masking_v1`
- Mask salt: `pawct-icassp2027-label-mask-v1`
- Official runnable train: 376 recordings and 285,674 eligible notes
- Composition-disjoint train: 355 recordings and 282,493 eligible notes
- The CLI is train-only, has no tunable mask/range/weight options, and scores
  the identical masked event IDs for RP, OC, and the fixed cyclic-range
  negative control.
- Divisi source notes remain distinct. No binary-head projection changes the
  evaluation denominator.
- Execution was CPU-only, single-threaded, low priority, and repeated twice.
  Each repeated JSON was byte-identical; the existing GPU process/session was
  unchanged.

## Results

| Training protocol | Mask | Masked notes | RP accuracy | RP macro F1 | OC accuracy | OC macro F1 | OC − RP macro F1 | RP − cyclic control | OC − cyclic control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Official runnable | 10% | 28,430 | 0.6420 | 0.6361 | 0.7409 | 0.7439 | +0.1079 | +0.4688 | +0.3453 |
| Official runnable | 25% | 71,399 | 0.6412 | 0.6367 | 0.7316 | 0.7351 | +0.0984 | +0.4694 | +0.3357 |
| Official runnable | 50% | 142,877 | 0.6413 | 0.6398 | 0.7196 | 0.7234 | +0.0835 | +0.4930 | +0.3053 |
| Composition-disjoint | 10% | 28,194 | 0.6325 | 0.6309 | 0.7371 | 0.7404 | +0.1095 | +0.4794 | +0.3483 |
| Composition-disjoint | 25% | 70,550 | 0.6325 | 0.6317 | 0.7243 | 0.7283 | +0.0966 | +0.4820 | +0.3392 |
| Composition-disjoint | 50% | 141,270 | 0.6354 | 0.6344 | 0.7130 | 0.7170 | +0.0826 | +0.4864 | +0.3011 |

`RP − cyclic control` compares aligned RP macro F1 with RP under cyclically
misassigned voice ranges; the OC column makes the analogous comparison. Full
per-voice scores, confusion matrices, exact event hashes, realized mask rates,
and visible-label ranges are retained in the JSON files.

The result supplies mechanism evidence for both priors: aligned RP remains
well above its range-mismatched control, while OC adds 0.083–0.110 macro F1
over RP across both training protocols and all mask rates. The gain decreases
as less trajectory context remains visible, which is directionally consistent
with the OC hypothesis. This diagnostic does **not** show that either prior
improves audio-to-note transcription; only matched P1b/P2/P3 acoustic runs can
support that claim.

## Reproduction

```bash
export YOUCHORALE_DATASET_DIR=/absolute/path/to/YouChorale
export YOUCHORALE_HDF5_DIR=/absolute/path/to/youchorale_sr16000
export SPLIT_DIR=repro/splits/youchorale_available_audio_434_composition_disjoint_v1

python tools/evaluate_label_masking.py \
  --dataset-dir "$YOUCHORALE_DATASET_DIR" --split train \
  --packed-hdf5-dir "$YOUCHORALE_HDF5_DIR" \
  --output-json official_train.json
python tools/evaluate_label_masking.py \
  --dataset-dir "$YOUCHORALE_DATASET_DIR" --split train \
  --recording-manifest "$SPLIT_DIR/train.json" \
  --packed-hdf5-dir "$YOUCHORALE_HDF5_DIR" \
  --output-json composition_train.json
```

## File integrity

```text
2f57e05c6f5c86f823c1584ce8de5877153e27c4481412b27da576ccfeef8f74  official_train.json
0c5ef22c078ad2d4c09d79bef93f235adaceddb7a2eef7d1596489a0a098799f  composition_train.json
```
