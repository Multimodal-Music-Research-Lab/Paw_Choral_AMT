# Source provenance

- Audit date: 2026-09-06 (Europe/London).
- Source: read-only snapshot from a private research workstation.
- Snapshot archive SHA-256:
  `52ce7300c73d66d38e5a2ec39596acbac5414e7ada566c400b610d74db2de39f`.
- Qualitative figure SHA-256:
  `46747902ae576b79ca3ec25e50cf2e2955aec66892a2ef2c8643012ce133dbc5`.
- Static syntax audit: all imported Python files parsed/compiled.
- Initial local dependency-backed release test: 35 passed, with one expected
  `mir_eval` resampling warning. The actively maintained suite is larger; use
  the current CI/test command rather than this historical count.
- Excluded from import: approximately 119 GB of HDF5 data, probabilities,
  checkpoints, logs, caches, and generated visualizations.

The source directory was not tracked by the surrounding historical Git
repository, so its parent commit is not a source revision for these files.
Timestamp correspondence and the exact Figure 3 command/example provide strong
evidence that this is the manuscript code family, but not an immutable
experiment snapshot.

## YouChorale target audit (2026-09-07)

The corrected target semantics at source commit
`abe243877b995b528c096c831eb3f884af2fb52d` were transferred as a Git bundle
with SHA-256
`68e4bc34adda24666657b23e404690f70c5a50e2ad108db5dca578c92cde1dd1`.
The audit ran on `lab5090` as a CPU-only, one-thread, low-priority process over
a trusted read-only dataset copy. It did not use the GPU, change the dataset,
or redistribute source annotations.

The resulting descriptive reports and interpretation are archived under
[`repro/audits/youchorale_targets_20260907_abe2438`](../repro/audits/youchorale_targets_20260907_abe2438/README.md).
They audit labels and target construction, not model accuracy. Report hashes:

```text
f69b70f6c906078fbeaff552e1a4ddd8695ae290b4e338b819a6efa8457e64f2  all.json
f269de17d192cbd42f9a700d9f4443210f5ed8b68e8d83576d4a3d3cc35384c1  test.json
d9f8962562fa24ea8b50b865fb8c53166537054117287afcbdda995dbdf0edf9  train.json
7b1d17fae61cbeb1b016a7e38488bde5e0cc9d12b2a15f9faf851cb42f36241b  validation.json
```

Packaging verification after importing the reports: 242 tests passed with two
documented dependency warnings, Python compilation succeeded, and the release
scan passed.

## Runnable HDF5 subset audit (2026-09-08)

Source commit `4f23e775ee2a18c0eed16505756973081d20d3ee` was transferred
in a bundle with SHA-256
`087cbd6747c124688cc2721a13f52f592b7069a82725a34fde455f9bfd761fa3`.
The same CPU-only/read-only procedure intersected the source manifests with the
recursive exact filename stems in the historical acoustic HDF5 directory.

The pack contains 434 of 452 manifest stems: 376/392 train, 28/30 validation,
and 30/30 test. Its sorted recording-stem set hashes to
`bedaf5f3a1b8d007217d7c448d4ed34bd968baf4eafc813864fa675b2dea333c`.
See the
[`available-audio-434` audit](../repro/audits/youchorale_packed_targets_20260908_4f23e77/README.md)
for target statistics, exact missing stems, scope limits, and report hashes.
Filename existence was checked; this audit does not claim a byte-level HDF5
content validation.

## Composition-disjoint split (2026-09-08)

The final split generator source commit is
`a6fa99bc9107d605f4e53057aed0a3ca896596fd`; its transfer bundle SHA-256 is
`945005bda68cf8bfdf42d2090190065eeb34d5f7183d8698f902eb977a79e712`.
The CPU-only run selected the same 434-ID set as the packed-target audit, ran
twice with byte-identical reports, and verified zero work overlap. The frozen
group map, 355/40/39 recording manifests, complete report, validation record,
and checksums are in the
[`composition-disjoint v1 bundle`](../repro/splits/youchorale_available_audio_434_composition_disjoint_v1/README.md).

An earlier isolated execution from `8ccdc41` stopped during external validation
because the split tool and target audit declared different sequence-hash
encodings. It did not alter data or GPU state and was not published. Commit
`a6fa99b` aligned the hash convention and added a regression test before the
final split was generated.

## Composition targets and RP/OC recovery (2026-09-08)

Source commit `bde12f65008d61b2e72872ea0345732d6ad9c1b5` was transferred
in a bundle with SHA-256
`3a72af72b6165470f836ae47c6aeb09be646c0928e746dd768f8e0658508dd11`.
A CPU-only, single-threaded, low-priority run audited all three frozen
composition-disjoint manifests and evaluated the train-only nested 10/25/50%
label-masking protocol on both the official runnable and
composition-disjoint training sets. Both label-masking reports were generated
twice and were byte-identical. The worker's existing GPU PID, command, working
directory, and persistent session were unchanged before and after the run.

The public artifacts are the
[`composition-disjoint target audit`](../repro/audits/youchorale_composition_disjoint_targets_20260908_bde12f6/README.md)
and the [RP/OC prior-recovery diagnostic](../repro/analyses/youchorale_prior_recovery_20260908_bde12f6/README.md).
They establish target/range provenance and prior mechanism behavior only;
neither is evidence of audio transcription performance.
