import pickle
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_DIR / 'src'
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from choral_targets import (
    ChoralTargetBuilder,
    require_complete_satb_reference,
    resolve_target_assignment,
    unlabelled_note_counts,
)


def make_cfg(
    target_assignment='part_name',
    *,
    legacy_assignment=None,
    preserve_known_part_labels=True,
):
    choral_values = {
        'num_voices': 4,
        'voice_names': ['S', 'A', 'T', 'B'],
        'preserve_known_part_labels': preserve_known_part_labels,
        'voice_assignment_part_penalty': 2.0,
        'voice_assignment_continuity_weight': 0.35,
        'voice_assignment_overlap_penalty': 4.0,
        'voice_assignment_range_mins': [60, 55, 48, 40],
        'voice_assignment_range_maxs': [88, 79, 72, 67],
        'voice_assignment_range_margin': 2.0,
        'voice_assignment_mask_penalty': 8.0,
        'oc_gap_decay_seconds': 2.0,
        'oc_overlap_tolerance_seconds': 0.05,
    }
    if target_assignment is not None:
        choral_values['target_assignment'] = target_assignment
    if legacy_assignment is not None:
        choral_values['voice_assignment_method'] = legacy_assignment
    return SimpleNamespace(
        feature=SimpleNamespace(
            begin_note=21,
            classes_num=88,
            frames_per_second=10,
            segment_seconds=2.0,
        ),
        choral=SimpleNamespace(**choral_values),
    )


def note(pitch, onset=0.0, offset=1.0):
    return [pitch, 0, 0, onset, offset]


class ResolveTargetAssignmentTest(unittest.TestCase):
    def test_canonical_config_has_priority_over_legacy_alias(self):
        cfg = make_cfg('oc', legacy_assignment='range_prior')
        self.assertEqual(resolve_target_assignment(cfg), 'ordered_continuity')

    def test_legacy_config_and_short_aliases_are_supported(self):
        self.assertEqual(
            resolve_target_assignment(make_cfg(None, legacy_assignment='rp')),
            'legacy_range_prior',
        )
        self.assertEqual(
            resolve_target_assignment(
                make_cfg(None, legacy_assignment='ordered_continuity')
            ),
            'legacy_ordered_continuity',
        )
        self.assertEqual(
            resolve_target_assignment(
                make_cfg(None, legacy_assignment='range_masked_continuity')
            ),
            'legacy_range_masked_continuity',
        )
        self.assertEqual(resolve_target_assignment(make_cfg('oc')), 'ordered_continuity')
        self.assertEqual(
            resolve_target_assignment(make_cfg('range_masked_continuity')),
            'range_masked_continuity',
        )
        self.assertEqual(
            resolve_target_assignment(make_cfg('legacy_range_prior')),
            'legacy_range_prior',
        )
        self.assertEqual(
            resolve_target_assignment(make_cfg('legacy_ordered_continuity')),
            'legacy_ordered_continuity',
        )
        self.assertEqual(
            resolve_target_assignment(make_cfg('legacy_range_masked_continuity')),
            'legacy_range_masked_continuity',
        )

    def test_invalid_assignment_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_target_assignment(make_cfg('not_a_method'))

    def test_explicit_builder_override_is_independent_of_training_assignment(self):
        builder = ChoralTargetBuilder(
            make_cfg('ordered_continuity'),
            target_assignment='part_name',
        )
        self.assertEqual(builder.target_assignment, 'part_name')


class ChoralTargetAssignmentTest(unittest.TestCase):
    def test_preserving_known_part_labels_is_the_default(self):
        cfg = make_cfg('range_prior')
        del cfg.choral.preserve_known_part_labels
        events = ChoralTargetBuilder(cfg).assign_voice_events([
            {'S': [note(45)]},
        ])
        self.assertEqual(events[0][0], 0)

    def test_rp_and_oc_retain_all_notes_and_preserve_known_labels(self):
        note_bars = [{
            'S': [note(45)],       # Intentionally outside the soprano prior.
            'unknown': [note(70)],
            'B': [note(80)],       # Intentionally outside the bass prior.
        }]

        for method in ('range_prior', 'ordered_continuity'):
            with self.subTest(method=method):
                events = ChoralTargetBuilder(make_cfg(method)).assign_voice_events(note_bars)
                by_pitch = {pitch: voice for voice, pitch, _, _ in events}
                self.assertEqual(len(events), 3)
                self.assertEqual(by_pitch[45], 0)
                self.assertEqual(by_pitch[80], 3)
                self.assertIn(by_pitch[70], range(4))

    def test_oc_preserves_known_divisi_without_forcing_four_distinct_voices(self):
        note_bars = [{
            'S1': [note(74)],
            'S2': [note(67)],
            'A': [note(69)],
            'T': [note(60)],
        }]
        events = ChoralTargetBuilder(
            make_cfg('ordered_continuity')
        ).assign_voice_events(note_bars)
        by_pitch = {pitch: voice for voice, pitch, _, _ in events}

        self.assertEqual(len(events), 4)
        self.assertEqual(by_pitch[74], 0)
        self.assertEqual(by_pitch[67], 0)
        self.assertEqual(by_pitch[69], 1)
        self.assertEqual(by_pitch[60], 2)
        self.assertNotIn(3, by_pitch.values())

    def test_quantized_unison_divisi_has_one_onset_and_latest_offset_target(self):
        cfg = make_cfg('part_name')
        cfg.feature.frames_per_second = 100
        note_bars = [{
            'S1': [note(72, onset=0.001, offset=0.5)],
            'S2': [note(72, onset=0.004, offset=1.0)],
            'A': [note(72, onset=0.001, offset=1.0)],
        }]

        builder = ChoralTargetBuilder(cfg)
        events = builder.assign_voice_events(note_bars)
        soprano_events = [event for event in events if event[0] == 0]
        alto_events = [event for event in events if event[0] == 1]
        self.assertEqual(soprano_events, [(0, 72, 0.001, 1.0)])
        self.assertEqual(alto_events, [(1, 72, 0.001, 1.0)])

        targets = builder.build(note_bars)
        soprano_note = 72 - cfg.feature.begin_note
        self.assertEqual(float(targets['voice_onset_roll'][:, 0, soprano_note].sum()), 1.0)
        self.assertEqual(float(targets['voice_offset_roll'][:, 0, soprano_note].sum()), 1.0)
        self.assertEqual(targets['voice_offset_roll'][100, 0, soprano_note], 1.0)

    def test_oc_retains_more_than_four_simultaneous_unknown_notes(self):
        note_bars = [{
            'unknown': [note(pitch) for pitch in (76, 72, 69, 65, 60)],
        }]
        events = ChoralTargetBuilder(
            make_cfg('ordered_continuity')
        ).assign_voice_events(note_bars)
        voice_counts = Counter(voice for voice, _, _, _ in events)

        self.assertEqual(len(events), 5)
        self.assertTrue(all(0 <= voice < 4 for voice in voice_counts))
        self.assertGreaterEqual(max(voice_counts.values()), 2)

    def test_oc_keeps_long_divisi_active_when_a_shorter_note_ends(self):
        note_bars = [{
            'S1': [note(72, onset=0.0, offset=10.0)],
            'S2': [note(60, onset=1.0, offset=2.0)],
            'unknown': [note(80, onset=3.0, offset=4.0)],
        }]

        events = ChoralTargetBuilder(
            make_cfg('ordered_continuity')
        ).assign_voice_events(note_bars)
        by_pitch = {pitch: voice for voice, pitch, _, _ in events}

        self.assertEqual(by_pitch[72], 0)
        self.assertEqual(by_pitch[60], 0)
        self.assertNotEqual(by_pitch[80], 0)

    def test_legacy_range_prior_relabels_known_notes(self):
        note_bars = [{'S': [note(45)]}]
        modern = ChoralTargetBuilder(
            make_cfg('range_prior')
        ).assign_voice_events(note_bars)
        legacy = ChoralTargetBuilder(
            make_cfg('legacy_range_prior')
        ).assign_voice_events(note_bars)

        self.assertEqual(modern[0][0], 0)
        self.assertEqual(legacy[0][0], 3)

    def test_legacy_oc_reproduces_all_note_reordering(self):
        note_bars = [{
            'S1': [note(76)],
            'S2': [note(72)],
            'S3': [note(69)],
            'S4': [note(65)],
        }]
        modern = ChoralTargetBuilder(
            make_cfg('ordered_continuity')
        ).assign_voice_events(note_bars)
        legacy = ChoralTargetBuilder(
            make_cfg('legacy_ordered_continuity')
        ).assign_voice_events(note_bars)

        self.assertEqual(Counter(voice for voice, _, _, _ in modern), {0: 4})
        self.assertEqual(Counter(voice for voice, _, _, _ in legacy), {0: 1, 1: 1, 2: 1, 3: 1})

    def test_builder_returns_existing_roll_mask_and_presence_interface(self):
        builder = ChoralTargetBuilder(make_cfg('range_prior'))
        base_mask = np.ones((21, 88), dtype=np.float32)
        base_mask[0, 0] = 0.0
        targets = builder.build(
            [{'S1': [note(72, onset=0.2, offset=1.2)]}],
            start_time=0.0,
            frame_mask_roll=base_mask,
            onset_mask_roll=base_mask,
            offset_mask_roll=base_mask,
        )

        self.assertEqual(targets['voice_frame_roll'].shape, (21, 4, 88))
        self.assertEqual(targets['voice_onset_roll'].shape, (21, 4, 88))
        self.assertEqual(targets['voice_offset_roll'].shape, (21, 4, 88))
        self.assertEqual(targets['voice_frame_mask_roll'].shape, (21, 4, 88))
        self.assertEqual(targets['voice_presence'].tolist(), [1.0, 0.0, 0.0, 0.0])
        for voice_idx in range(4):
            np.testing.assert_array_equal(
                targets['voice_frame_mask_roll'][:, voice_idx, :],
                base_mask,
            )

    def test_same_voice_overlap_is_projected_to_rearticulation_boundary(self):
        builder = ChoralTargetBuilder(make_cfg('part_name'))
        events = builder.assign_voice_events([{
            'S1': [
                note(72, onset=0.0, offset=1.1),
                note(72, onset=1.0, offset=2.0),
            ]
        }])

        self.assertEqual(
            events,
            [(0, 72, 0.0, 1.0), (0, 72, 1.0, 2.0)],
        )

    def test_formal_reference_guard_reports_unlabelled_notes(self):
        note_bars = [{
            'S1': [note(72)],
            'mystery_part': [note(67), note(65)],
            'measure': 1,
        }]

        self.assertEqual(unlabelled_note_counts(note_bars), {'mystery_part': 2})
        with self.assertRaisesRegex(RuntimeError, 'mystery_part'):
            require_complete_satb_reference(note_bars, 'song.pkl')

    def test_formal_reference_guard_rejects_unrepresentable_or_malformed_notes(self):
        with self.assertRaisesRegex(RuntimeError, 'outside_model_range'):
            require_complete_satb_reference(
                [{'S': [note(20)]}],
                'low-note.pkl',
                begin_note=21,
                classes_num=88,
            )
        with self.assertRaisesRegex(RuntimeError, 'schema_errors'):
            require_complete_satb_reference(
                [{'S': tuple([note(72)])}],
                'tuple-container.pkl',
                begin_note=21,
                classes_num=88,
            )

    def test_formal_reference_guard_rejects_pickled_list_iterator(self):
        note_iterator = pickle.loads(pickle.dumps(iter([{'S': [note(72)]}])))
        with self.assertRaisesRegex(RuntimeError, 'top_level_schema=list_iterator'):
            require_complete_satb_reference(note_iterator, 'iterator.pkl')


if __name__ == '__main__':
    unittest.main()
