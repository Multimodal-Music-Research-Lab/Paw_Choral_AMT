# Reproducibility status

This is an evidence-based status page for the audited research snapshot. A
checked item means the repository itself supports the claim; it does not mean
the manuscript's tables have been rerun.

## Present

- [x] PawCT shared encoder, SATB heads, presence gate, and max union output.
- [x] Part-name, range-prior, and ordered-continuity target assignment.
- [x] Per-voice, union, and presence losses.
- [x] PagCT and note-level BiLSTM Post-VA implementations.
- [x] Merged and SATB evaluation at 50/100 ms onset tolerance.
- [x] Manuscript Figure 3 generation path and matching example identifier.
- [x] Validation/test probability separation and validation-default threshold search.
- [x] Public paths, declared dependencies, upstream attribution, and release scan.
- [x] Data-free PawCT forward/union/loss-backward smoke test.

## Missing before a reproducibility claim

- [ ] Immutable source commit corresponding to every reported run.
- [ ] Exact expanded Hydra configuration and command for each table row.
- [ ] Composition-disjoint split manifest with hashes.
- [ ] Offline seven-transposition generation code and alignment tests.
- [ ] Checkpoint files with SHA-256, config, environment, and load-key audit.
- [ ] Validation-loss early stopping or corrected manuscript description.
- [ ] A single command that regenerates Tables 1 and 2 within a tolerance.
- [ ] VA/retention metric code tied to the exact numerator and denominator.
- [ ] Direct audio-to-four-track SATB MIDI CLI.
- [ ] PagCT parity, data-integrity, and real-data integration tests.
- [ ] Multi-seed or composition-level bootstrap uncertainty.

## Located historical evidence

The private workstation audit found a source snapshot archive dated 2026-04-20
with SHA-256
`52ce7300c73d66d38e5a2ec39596acbac5414e7ada566c400b610d74db2de39f`.
The live source added Figure 3 and predicted-MIDI utilities through 2026-04-28,
the manuscript PDF's creation date. The source directory itself was untracked
inside its parent Git repository.

Historical acoustic logs and visualization commands point to 299999-step
checkpoints. The manuscript states 200k iterations and validation-loss early
stopping. Historical Post-VA arguments record 80 epochs, batch size 32, AdamW,
learning rate `1e-3`, presence weight `0.1`, range weight `0.02`, and crossing
weight `0.05`; the manuscript describes a different optimization setup.

No exact local score artifact located during audit reproduced every Table 2
number. For example, one PawCT-OC evaluation summary averaged approximately
0.219 note F1 while the manuscript reports 0.225. This may be a checkpoint,
threshold, or evaluator-version difference and must be traced rather than
rounded away.

## Scientific alignment issues

1. **PagCT/PawCT capacity:** PagCT has three complete acoustic branches;
   PawCT has one shared encoder. Report parameters/FLOPs or implement a matched
   counterpart.
2. **Onset/offset regression:** regression rolls are generated but are not used
   by the located loss, while regression decoding still estimates sub-frame
   positions.
3. **Post-VA feature shift:** beat-related fields used in ground-truth-note
   training are zero-filled for some predicted-note evaluation paths.
4. **Presence metric:** most recordings contain all four parts, so compare the
   head with an always-on baseline and report performance on missing-part
   examples.
5. **Threshold provenance:** never select decoding thresholds on test; this
   release includes a guard, but historical threshold files need provenance.
6. **Claim wording:** without paired confidence intervals or multiple seeds,
   avoid “significantly.”

## Required release bundle per model

```text
checkpoints/<model>/
├── model.safetensors or model.pth
├── config.resolved.yaml
├── environment.txt
├── source_commit.txt
├── split_manifest.json
├── metrics.json
└── SHA256SUMS
```

If a legacy `.pth` is published, document that it must be treated as trusted
code-bearing input and report missing/unexpected state-dict keys at load time.
