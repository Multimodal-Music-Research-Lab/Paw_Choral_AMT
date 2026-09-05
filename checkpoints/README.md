# Checkpoints

No model weights are included in this audited snapshot. Before publishing a
checkpoint, confirm redistribution authority and include:

- the exact resolved configuration and source commit;
- dataset and split-manifest hashes;
- training/evaluation environment;
- reported metrics and threshold provenance;
- SHA-256 for every artifact;
- missing/unexpected state-dict keys (expected to be empty).

Do not commit optimizer state or absolute workstation paths unless they are
necessary and intentionally scrubbed.
