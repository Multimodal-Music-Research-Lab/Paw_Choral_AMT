# PawCT/PagCT ICASSP Experiment Plan

This document separates verified audit findings from experiments that still
need to be run. It is deliberately stricter than the manuscript draft: a test
score is reported only after the configuration, checkpoint rule, decoder, and
thresholds have been frozen on validation data.

## 1. What the current evidence says

The paper-reported ablation has a useful but incomplete signal:

- PawCT-RP raises mean frame F1 from 0.458 to 0.510, while mean note F1 falls
  from 0.217 to 0.209.
- PawCT-OC reaches the best reported mean note F1, 0.225, but the gain over
  PawCT is only 0.008 and has no confidence interval or multi-seed estimate.

The code/data audit found six confounds that must be fixed before interpreting
those differences. The descriptive counts below come from the
[runnable-pack target audit](../repro/audits/youchorale_packed_targets_20260908_4f23e77/README.md),
not from a model-performance run:

1. The manuscript describes 452 recordings, but the located acoustic pack has
   434: 376/392 train, 28/30 validation, and 30/30 test. The missing 18 items
   have annotations but no local supported audio. A new result must either
   recover/repack them or explicitly use the frozen `available-audio-434`
   protocol; silent subset selection is invalid.
2. Modern anchored RP and OC each change exactly 0 of 285,674 runnable training
   labels because every in-range training note has a canonical
   SATB/divisi-derived label. Their effect on YouChorale must therefore come
   from a soft output prior, not hard relabeling.
3. Historical/legacy RP relabels 761 of 285,674 runnable training notes
   (0.2664%), so it is too weak as a hard target transformation to support a
   large note-level effect.
4. Historical/legacy OC relabels 41,347 of 285,674 runnable training notes
   (14.4735%).
   YouChorale includes S1/S2, A1/A2, and other divisi labels, so forcing a
   one-to-one SATB assignment can overwrite valid part labels. Modern anchored
   RP/OC does not relabel these trusted notes.
5. Stored RP/OC target rolls were used as frame-level evaluation references,
   whereas note metrics used original part names. Frame and note scores were
   therefore not measuring the same voice definition.
6. The current onset/offset heads are trained against one-frame binary targets,
   while the documented regression decoder expects smooth regression targets.

These are audit observations, not new experimental results.

## 2. Main scientific question

The clean ICASSP question is:

> Can a shared acoustic representation preserve the merged transcription while
> producing stable, part-aware SATB trajectories under ambiguous or incomplete
> part supervision?

PagCT answers the merged-note question. PawCT adds explicit SATB heads. RP and
OC should be treated as priors for ambiguous supervision or regularization,
not as permission to overwrite trusted canonical labels.

## 3. Pre-registered comparison

All PawCT ablations use the same data, augmentation, maximum update budget,
stopping rule, checkpoint-selection proxy, decoder family, and seed set.
Cross-family baselines use the same split, seeds, and validation/test freeze,
but retain a model-appropriate checkpoint-selection metric.

Until the missing audio is restored, every current pilot uses the frozen
`available-audio-434` intersections and their committed ID hashes. If a
complete 452-recording pack is later built, it becomes a separately named
protocol and every compared row must be retrained; rows from the two protocols
cannot be combined in one ablation table.

Every `row × hyperparameter setting × seed` must use a unique run suffix, and
the same suffix must be propagated through training, inference, threshold
search, and scoring. Otherwise a later seed can overwrite `best.pth`, numeric
checkpoints, logs, or probability artifacts from an earlier seed.

| ID | System | Trusted labels | Prior / regularizer | Purpose |
| --- | --- | --- | --- | --- |
| P0 | PagCT | merged | none | part-agnostic reference |
| P1a | PawCT-no-union | canonical part names | no union, RP, or OC | isolate union contribution |
| P1b | PawCT | canonical part names | frame/onset union only | main part-aware baseline |
| P2 | PawCT-RP | canonical labels anchored | train-only p01/p99 range penalty on unsupported outputs | test register prior |
| P3 | PawCT-RP+OC | canonical labels anchored | frozen P2 plus target-relative pitch-trajectory consistency | isolate trajectory-prior increment |
| P4 | PagCT + Post-VA | canonical symbolic labels | post-hoc assignment | two-stage reference; exploratory until feature parity is fixed |

Legacy RP/OC, which relabel every known note, may be retained only as a clearly
named diagnostic row. They should not be the primary proposed methods.

All corrected rows use the same frame-quantized canonical union projector over
the source `note.pkl` intervals. Scoring tolerance never changes this reference.
Because this repairs merged-MIDI overwrite, long-note backtracking, and boundary
mask errors, historical checkpoints and their reported `0.225` OC value cannot
be mixed with corrected rows; every main contrast needs a clean retraining run.

### Staged compute plan

1. **Zero-training audit:** checkpoint-key compatibility, target retention,
   decoder boundary cases, and immutable evaluation references.
2. **Existing-checkpoint decoder study:** decode the same validation
   probabilities with the binary onsets-and-frames decoder and the regression
   decoder. This isolates decoding from training.
3. **Short pilot:** seed 86 and exactly 40,000 updates for P1b/P2/P3.
   Advance only variants that improve the pre-registered validation objective
   without materially reducing merged recall.
4. **Main run:** seeds 17, 42, and 86, a 200,000-update maximum, identical
   evaluation frequency and early-stopping patience, validation-selected
   checkpoint and thresholds, then one locked test pass. Optional extra seeds
   are reported as a separate sensitivity analysis, not added after seeing the
   test result.

For all PawCT rows, freeze complete-validation `mean_voice_frame_ap` as the
threshold-free checkpoint-selection proxy. It is not the primary endpoint;
macro SATB note F1 is computed after thresholds are chosen on validation. This
separation must be stated explicitly and kept identical across rows.

The pilot search is sequential and fixed before test evaluation. P2 and P3 use
the train-only p01/p99 MIDI ranges `mins=[60,55,50,41]` and
`maxs=[79,74,70,62]` for S/A/T/B, with a two-semitone margin. These values come
from the 376-recording runnable training subset, not from missing-audio
annotations. The loss excludes
trusted annotated positives, so genuine out-of-register notes are not punished.
The older broad defaults `[60,55,48,40]..[88,79,72,67]` are retained as one
named diagnostic ablation, not added to the primary grid:

1. P1b ties frame/onset union weights and searches `{0.10, 0.25, 0.50}`.
2. P2 fixes the P1b winner and the train-derived range, then searches RP weight
   `{0.003, 0.01, 0.03}`.
3. P3 fixes the P1b and P2 winners, including the exact RP range/margin, then
   searches OC weight `{0.003, 0.01, 0.03}` at the pre-registered two-second
   silent-gap decay. If the winning OC candidate passes the promotion gate,
   compare `0.5` and `8.0` seconds at that one frozen weight as a second-stage
   mechanism ablation. This sequential design separates weight from decay and
   avoids six unnecessary crossed pilot runs before the deadline.

For each candidate, select thresholds on validation and rank by 50 ms macro
SATB note F1, subject to union recall falling by at most 0.01 relative to its
parent row. Ties are resolved by higher union F1, then lower regularizer
weight, then lexicographic configuration ID. Repeat this frozen procedure
separately for the official and composition-disjoint protocols. PagCT selects
`frame_ap`; Post-VA selects validation macro assignment F1. P4 is not eligible
for a confirmatory cross-system claim until its train/evaluation feature
parity is repaired and tested.

## 4. Primary and secondary metrics

The primary metric is macro-average SATB note F1 at 50 ms onset tolerance,
computed against original canonical part labels. Report per-composition mean,
paired differences, and a 95% composition-level bootstrap confidence interval.

Secondary metrics:

- SATB note F1 at 100 ms and note-with-offset F1;
- merged/union note precision, recall, and F1 from the same PawCT output;
- per-part S/A/T/B note F1 and the minimum-part F1;
- frame F1 against an immutable part-name reference;
- matched-note voice accuracy and the 4-by-4 voice confusion matrix;
- note fragmentation, duplicate-note rate, voice-switch rate, and range
  violations among false positives;
- parameter count, FLOPs or MACs, and real-time factor;
- presence AUPRC and recall on the full set when both classes occur, plus
  false-positive rate and specificity on actually absent parts, compared with
  an always-SATB baseline.

Do not call the old `VA Rate` a voice-assignment accuracy. If retained, name it
`note retention ratio` and keep its denominator explicit.

## 5. Analyses beyond the two manuscript tables

### A. Prior validity / label-recovery study

Mask a controlled 10%, 25%, and 50% of known training part labels
and ask RP/OC to recover them. Report accuracy, macro-F1, and confusion by part.
This directly tests whether the priors contain useful assignment information
without corrupting ground truth.

The committed `tools/evaluate_label_masking.py` protocol deliberately narrows
this to training data: nested SHA-256 masks hold out 10/25/50% of source notes;
RP ranges are re-estimated only from the still-visible labels; OC may use those
visible labels as within-recording temporal context; and only masked notes enter
the denominator. A fixed cyclic mapping of voice ranges is evaluated on the
identical held-out IDs as a negative control. The tool preserves individual
divisi notes rather than projecting them into binary-head events and exposes no
validation/test mode or tunable mask/weight CLI.

### B. Difficulty-stratified transcription

Stratify pieces or notes by:

- local polyphony (1, 2, 3, 4, greater than 4);
- pitch-range overlap between adjacent parts;
- crossing/unison/divisi versus ordinary passages;
- note duration and onset density;
- seen composition versus composition-disjoint evaluation.

This should reveal where OC helps: its expected benefit is in ambiguous,
temporally connected passages, not isolated easy notes.

The runnable-pack audit gives a concrete reason to pre-register these strata:
greater-than-four-note onset groups account for 14.60% of train, 17.81% of
validation, and 16.17% of test groups; duplicate-canonical-voice groups account
for 33.72%, 43.99%, and 38.80%, respectively. These descriptive differences do
not prove an OC performance effect, but they show that one aggregate mean can
hide materially different divisi/unison/polyphony conditions.

The corrected OC regularizer compares predicted pitch motion with annotated
pitch motion between consecutive annotated onset events, including transitions
across rests. A correct leap has zero trajectory-residual loss; a voice swap or
unsupported jump does not. Event-level normalization prevents long held notes
from washing out the transition, while exponential gap decay reduces confidence
in distant links. This removes the former bias toward flat contours and makes
`RP+OC - RP` a cleaner test of temporal structure.

### C. Oracle decomposition

Compare (i) ground-truth notes plus Post-VA, (ii) PagCT notes plus oracle voice,
(iii) PagCT plus Post-VA, and (iv) PawCT. The four rows separate acoustic note
errors from voice-assignment errors and show whether an end-to-end model is
limited by detection or assignment.

### D. Architecture ablations

- presence head as auxiliary-only versus probability gate;
- max-union frame/onset weight in {0, 0.1, 0.25, 1.0};
- max-derived union offset disabled versus a dedicated learned union-offset
  head (the max of voice releases is not a valid union release under
  same-pitch overlap);
- max union versus a separately learned union head;
- OC alone versus RP+OC, while the confirmatory OC contrast remains RP+OC
  versus RP so continuity is the only changed factor;
- binary onset/offset training plus binary decoder versus genuine regression
  targets/loss plus regression decoder;
- 10 s independent windows versus a longer-context or state-carrying model.

## 6. Split and statistical protocol

The official recording split has substantial composition overlap between
training and evaluation recordings. Using exact normalized `(composer, title)`
keys from the public `info.csv`, 22 of the 29 test works also occur in train;
23 of 30 test recordings therefore come from a train-seen work. Keep this
recording split for comparison with prior work, but add a composition-disjoint
split as the stronger generalization result. Publish the exact manifests and
grouping rule, and report both rows without conflating them.

For the current `available-audio-434` subset, train contains 229 works,
validation 27, and test 29. Twenty-one of 29 test works occur in the runnable
train set, and 22/30 test recordings come from a train-seen work. These figures
must be recomputed from the frozen manifest if missing audio is restored.

For the disjoint protocol, normalize `(composer, title)` with Unicode NFKC,
case-folding, punctuation removal, and collapsed whitespace; keep every
recording of a normalized work in one group. Sort groups by
`SHA256("pawct-icassp2027-v1" + group_key)`, then assign the first 80% of groups
to train, the next 10% to validation, and the remainder to test (integer cuts
use floor). Generate the report, group map, and manifests with
`tools/build_youchorale_composition_split.py`; the tool records the Unicode
database version and sequence-hash encoding, verifies zero generated work
overlap, and refuses to overwrite a different or partial frozen output.
The resulting v1 protocol is frozen in
[`repro/splits/youchorale_available_audio_434_composition_disjoint_v1`](../repro/splits/youchorale_available_audio_434_composition_disjoint_v1/README.md):
355/40/39 recordings and 193/24/25 groups in train/validation/test, with zero
pairwise work overlap.
Retrain every row on this train split, make every model/threshold choice on its
validation split, and do not inspect its test metrics until the run registry is
frozen. Only runs that consume these exact manifests may use the label
“composition-disjoint v1.”

Pass
`dataset.youchorale_split_dir=./repro/splits/youchorale_available_audio_434_composition_disjoint_v1`
to training, validation inference, test inference, and scoring, together with
`exp.name_suffix=composition_disjoint_v1` to isolate checkpoints and
probabilities from the official-split run. The loader verifies the three-way
manifest partition against the entire packed stem set before reading a batch.

For every selected comparison:

- use identical seeds and paired per-composition deltas;
- report mean, standard deviation, and 95% bootstrap CI;
- bootstrap at the composition level, not at the note level;
- use Holm correction when making several confirmatory pairwise claims;
- reserve the test split for one final evaluation after validation choices are
  frozen.
- archive a complete split manifest and require a single checkpoint SHA-256
  across every probability artifact before computing any table entry.

For aggregation, pool matched-note counts across recordings of the same work
within each voice, compute four voice F1 values, then macro-average S/A/T/B.
Average works equally within a seed and report mean ± standard deviation across
the three seeds. For a paired contrast, first average each work's paired delta
over the matched seeds, then compute a 10,000-resample paired BCa bootstrap over
works. This confidence interval is conditional on the fixed seed set; seed
variation is reported separately.

Pre-register the confirmatory contrasts `P1b > P1a` and `P3 > P2`. Promote
`P1b > P4` to confirmatory only after the Post-VA feature-parity gate above is
satisfied; otherwise report it as exploratory. Call OC an overall
improvement only if the paired composition-level 95% CI for `P3 - P2`
lies above zero (with a practical target of at least `+0.005` note F1), union
F1 falls by no more than `0.005`, and union recall falls by no more than
`0.01`. A benefit confined to overlap/crossing/unison passages should be
reported as a targeted benefit, not a global gain.

## 7. Recommended paper flow

1. Mixed choral audio requires both accurate note detection and stable part
   identity.
2. PagCT establishes a strong merged-note baseline.
3. PawCT shares acoustic evidence across four explicit part streams while a
   union objective protects merged transcription.
4. Every audited note already has a canonical label, so hard assignment adds no
   supervision; the legacy OC rule is additionally unsafe because it overwrites
   14.47% of trusted runnable training labels.
5. Anchored RP keeps trusted notes while suppressing unsupported output mass
   outside a train-derived register using negative-label BCE; the current OC
   loss adds target-relative onset-event trajectory consistency, including
   across rests with gap decay. Multi-pitch/divisi onset frames remain under
   full BCE supervision but act as trajectory barriers rather than being
   reduced to fictitious centroids.
   Gap-aware/divisi-safe assignment applies only when a training label is
   genuinely unknown and must not be claimed as the main loss.
6. Overall, difficulty-stratified, label-recovery, and oracle analyses explain
   when the priors help and where remaining errors originate.

Avoid “significant” or “state of the art” unless the final multi-seed,
composition-level analysis supports those claims.

## 8. Methodological references

- YouChorale metadata and official split manifests:
  [info.csv](https://github.com/ella-granger/YouChorale/blob/main/info.csv),
  [train.json](https://github.com/ella-granger/YouChorale/blob/main/train.json),
  [test.json](https://github.com/ella-granger/YouChorale/blob/main/test.json).
- Note and note-with-offset metric definitions:
  [mir_eval transcription](https://mir-eval.readthedocs.io/latest/api/transcription.html).
- Voice-assignment cues and temporal continuity:
  [McLeod et al.](https://apmcleod.github.io/pdf/Vocal4-ismir.pdf) and
  [multi-trajectory modelling](https://arxiv.org/abs/2304.14848).
- Crossing, convergence, and overlap corner cases:
  [Gray and Bunescu](https://arxiv.org/abs/2011.03028).
- Clustered confidence intervals should follow a documented bootstrap
  procedure; the BCa method is described by
  [Efron (1987)](https://doi.org/10.1080/01621459.1987.10478410).
