# Reproduction contract

The CSV files under `expected/` are machine-readable transcriptions of the
manuscript's reported tables. They are reference targets, not measurements
produced by this commit.

The JSON files under
[`audits/youchorale_targets_20260907_abe2438`](audits/youchorale_targets_20260907_abe2438/README.md)
are descriptive real-data and target-construction audits. They establish label
coverage, onset-group structure, pitch statistics, and modern-versus-legacy
RP/OC behavior; they do **not** reproduce Table 1 or Table 2 and contain no
model predictions. Reports are split into
[`train`](audits/youchorale_targets_20260907_abe2438/train.json),
[`validation`](audits/youchorale_targets_20260907_abe2438/validation.json),
[`test`](audits/youchorale_targets_20260907_abe2438/test.json), and
[`all`](audits/youchorale_targets_20260907_abe2438/all.json).

The newer
[`audits/youchorale_packed_targets_20260908_4f23e77`](audits/youchorale_packed_targets_20260908_4f23e77/README.md)
intersects those manifests with the historical acoustic HDF5 stems. It is the
authoritative target/range audit for runs using that 434-recording pack and
documents its incomplete train/validation coverage. It remains descriptive,
not a model result.

Future corrected model measurements must live in a separately named locked
result bundle with source, configuration, checkpoint, split, and artifact
hashes. They must never overwrite either the historical `expected/` values or
the descriptive `audits/` evidence.

A complete release should provide a `reproduce_tables.py` command that reads a
locked test result bundle, regenerates both CSVs, and fails if a metric differs
from the frozen expected value beyond a declared tolerance.
