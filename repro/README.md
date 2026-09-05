# Reproduction contract

The CSV files under `expected/` are machine-readable transcriptions of the
manuscript's reported tables. They are reference targets, not measurements
produced by this commit.

A complete release should provide a `reproduce_tables.py` command that reads a
locked test result bundle, regenerates both CSVs, and fails if a metric differs
from the frozen expected value beyond a declared tolerance.
