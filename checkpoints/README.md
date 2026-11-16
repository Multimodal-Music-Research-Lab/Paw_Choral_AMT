# Checkpoints

[Download PagCT and PawCT base from Hugging Face](https://huggingface.co/Hanyu-Meng/PawCT-PagCT).
See the [checkpoint table](../README.md#checkpoints) for direct downloads,
inference configurations, and decoding thresholds.

Before publishing a checkpoint, confirm redistribution authority and include
all of the following.

- the exact resolved configuration and source commit;
- dataset and split-manifest hashes;
- training/evaluation environment;
- reported metrics and threshold provenance;
- SHA-256 for every artifact;
- missing/unexpected state-dict keys (expected to be empty).

Do not commit optimizer state or absolute workstation paths unless they are
necessary and intentionally scrubbed.
