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
