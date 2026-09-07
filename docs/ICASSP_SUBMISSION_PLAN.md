# ICASSP 2027 submission plan

This is a decision document, not a result report. It records the paper story,
evidence gates, compute order, and page budget before corrected test results are
available.

## External constraints

- The [official call for papers](https://2027.ieeeicassp.org/call-for-papers/)
  lists **16 September 2026** as the full-paper deadline.
- The [official author guidelines](https://2027.ieeeicassp.org/author-guidelines/)
  allow four pages of technical content and an optional fifth page containing
  references only. They also require disclosure in the acknowledgments when AI
  generated substantive article content such as text, figures, or code; simple
  editing or grammar assistance is treated separately.
- The [official editorial policy](https://2027.ieeeicassp.org/about/editorial-policies/)
  explicitly evaluates importance, novelty, technical correctness,
  experimental validation, clarity, and relation to prior work. The same page
  encourages supporting code and data for reproducibility.

The submission should therefore make one narrow, well-supported claim rather
than carry every historical experiment into a four-page paper.

## One-sentence claim

> Coupling shared SATB heads to a merged-note objective preserves acoustic note
> content, while label-anchored register and target-relative trajectory priors
> can reduce unsupported voice errors without rewriting trusted part labels.

The word “can” must be replaced by a stronger claim only if the locked
multi-seed comparison supports it. RP is not required to raise aggregate note
F1: a reproducible reduction in out-of-range false positives or improved
calibration without a material recall loss is a valid, narrower result. OC is
an overall improvement only if the paired composition-level confidence
interval for `P3 - P2` excludes zero; otherwise describe a targeted benefit in
pre-registered crossing/divisi/high-polyphony strata, or a negative result.

## Evidence hierarchy

### Submission-critical

1. Corrected P1b/P2/P3 runs on one frozen protocol with identical seeds,
   budgets, checkpoint selection, decoder, and validation-only thresholds.
2. Three seeds for the primary `P3 - P2` contrast and a paired bootstrap over
   composition groups.
3. Per-part note F1, merged-note precision/recall/F1, matched-note voice
   confusion, range-violation false positives, and voice-switch rate—not only
   a single mean.
4. Exact split/config/checkpoint/probability hashes and a one-command table
   builder.
5. One union ablation (`P1a` versus `P1b`) under the same corrected pipeline.

### Strong if time permits

1. Repeat selected P1b/P2/P3 settings on the frozen composition-disjoint v1
   protocol. A one-seed row must be labelled preliminary; it cannot substitute
   for uncertainty on the primary protocol.
2. Controlled 10/25/50% label-masking recovery for RP/OC, including the fixed
   cyclic-range negative control implemented by the frozen diagnostic.
3. Oracle decomposition: ground-truth notes + assignment, detected notes +
   oracle voices, detected notes + repaired Post-VA, and end-to-end PawCT.
4. Parameter count, MACs, and real-time factor for a capacity/fairness check.

### Supplementary or follow-up

- presence gating, learned union-offset, long-context/state carry, every
  architecture variant, and a full Post-VA redesign;
- any comparison whose feature parity, checkpoint provenance, or test-free
  threshold selection cannot be demonstrated.

## Evidence ledger as of 8 September 2026

| Evidence | Status | What it supports |
| --- | --- | --- |
| Frozen `available-audio-434` split and runnable-pack audit | Complete | Honest data denominator and official-split leakage disclosure |
| Frozen composition-disjoint v1 manifests and target audit | Complete | Zero work overlap, split-specific RP ranges, difficulty pre-registration |
| Train-only 10/25/50% RP/OC label recovery with cyclic control | Complete | Both priors encode assignment information; OC adds temporal-context value |
| Corrected P1a/P1b/P2/P3 acoustic pilots | Pending GPU availability | Validation-only selection of union/RP/OC weights |
| Three-seed locked P1b/P2/P3 comparison and bootstrap | Pending pilot gate | Main transcription claim |

The completed label-recovery diagnostic is mechanism evidence only. On the
composition-disjoint train set, RP obtains `0.631–0.634` macro F1 and OC
`0.717–0.740`; OC exceeds RP by `0.083–0.110`, while both beat their identical-ID
cyclic-range controls. These values must not be placed in the acoustic
transcription table. Exact results and hashes are in the
[versioned artifact](../repro/analyses/youchorale_prior_recovery_20260908_bde12f6/README.md).

## Compute order and stop gates

Historical 5090 logs suggest approximately 14.5–15.5 minutes per 5,000
training updates plus about one minute for validation. Treat this only as a
scheduling estimate: roughly 2.1 hours for a 40k pilot and 10.5 hours for a
200k run, to be replaced by timing from the first corrected run.

1. Run a CPU/data preflight and a short GPU smoke test after the worker is
   genuinely free.
2. Pilot seed 86 for 40k updates. Tune sequentially, without test access:
   union weight `{0.10, 0.25, 0.50}`; then RP weight
   `{0.003, 0.01, 0.03}`; then OC weight `{0.003, 0.01, 0.03}` at two-second
   gap decay. Only if OC survives its gate, compare decay `0.5` and `8.0`
   seconds at the frozen winning weight.
3. Reject a candidate if validation macro SATB note F1 does not improve under
   the registered tie-breaker or merged recall drops by more than 0.01 from its
   parent.
4. Continue the winning seed-86 checkpoint and launch seeds 17/42 for
   P1b/P2/P3. Run the P1a union ablation next.
5. After configurations and thresholds are frozen, perform one test pass and
   lock the result bundle. Composition-disjoint runs follow only after the
   primary bundle is safe.

Do not add hyperparameters after inspecting test outcomes. If time runs short,
drop lower-priority rows rather than shorten or otherwise change only the
unfavorable configurations.

## Four-page paper shape

| Space | Content |
| --- | --- |
| 0.5 page | Problem, why merged AMT is insufficient, and the one-sentence contribution |
| 1.1 pages | Shared PawCT encoder/heads, union objective, anchored RP, and target-relative OC |
| 0.7 page | `available-audio-434` disclosure, official versus composition-disjoint protocol, metrics, seeds, and statistics |
| 1.3 pages | One compact main table plus one mechanism/error-analysis figure |
| 0.4 page | Limitations and conclusion |

The main table should prioritize P1a/P1b/P2/P3 and include SATB note F1,
merged recall/F1, range-violation FP rate, and voice-switch rate. The single
figure should show paired OC-minus-RP effects by difficulty stratum with
confidence intervals, or a voice-confusion comparison if the stratified sample
is too small. Historical manuscript values belong in prose as unverified prior
observations, not beside corrected values in the same numeric block.

## Method-to-code consistency gate

The former manuscript equations describe hard reassignment, but every runnable
YouChorale training note already has a canonical label. The corrected method
section must instead describe:

1. immutable canonical SATB/divisi supervision;
2. a merged-note union objective derived from the same source notes;
3. an output-space register penalty that excludes trusted positives; and
4. an onset-event, target-relative continuity loss with explicit gap handling.

If the final code changes any of these semantics, increment the checkpoint
semantic version, update the method text, and retrain all affected rows. Never
reuse a historical checkpoint merely because its parameter keys still load.
