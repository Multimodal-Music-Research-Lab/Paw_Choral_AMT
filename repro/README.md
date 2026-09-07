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

Future corrected model measurements must live in a separately named locked
result bundle with source, configuration, checkpoint, split, and artifact
hashes. They must never overwrite either the historical `expected/` values or
the descriptive `audits/` evidence.

A complete release should provide a `reproduce_tables.py` command that reads a
locked test result bundle, regenerates both CSVs, and fails if a metric differs
from the frozen expected value beyond a declared tolerance.
