import hashlib
import json
import unittest
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
SPLIT_DIR = (
    REPO_DIR
    / 'repro'
    / 'splits'
    / 'youchorale_available_audio_434_composition_disjoint_v1'
)
SPLIT_NAMES = ('train', 'valid', 'test')


def sequence_sha256(values):
    payload = ''.join(f'{value}\n' for value in sorted(values)).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


class FrozenCompositionSplitTest(unittest.TestCase):
    def test_frozen_manifests_match_report_and_group_map(self):
        report = json.loads((SPLIT_DIR / 'audit_report.json').read_text())
        validation = json.loads((SPLIT_DIR / 'validation.json').read_text())
        group_map = json.loads((SPLIT_DIR / 'group_map.json').read_text())
        manifests = {
            name: json.loads((SPLIT_DIR / f'{name}.json').read_text())
            for name in SPLIT_NAMES
        }

        split_report = report['composition_disjoint_split']
        self.assertEqual(
            split_report['recording_counts'],
            {'train': 355, 'valid': 40, 'test': 39},
        )
        self.assertEqual(
            split_report['group_counts'],
            {'train': 193, 'valid': 24, 'test': 25},
        )
        self.assertEqual(report['selection']['recordings'], 434)
        self.assertEqual(report['selection']['groups'], 242)
        self.assertTrue(
            all(
                pair['shared_groups'] == 0
                for pair in split_report['work_overlap'].values()
            )
        )
        self.assertTrue(validation['rerun_report_byte_identical'])

        all_ids = set()
        for name, recording_ids in manifests.items():
            self.assertEqual(len(recording_ids), len(set(recording_ids)))
            self.assertFalse(all_ids & set(recording_ids))
            all_ids.update(recording_ids)
            self.assertEqual(
                sequence_sha256(recording_ids),
                split_report['recording_ids_sha256'][name],
            )
        self.assertEqual(len(all_ids), 434)
        self.assertEqual(set(group_map['recordings']), all_ids)
        self.assertEqual(
            sequence_sha256(all_ids),
            report['selection']['recording_ids_sha256'],
        )

        for name, expected_hash in split_report['artifact_sha256'].items():
            self.assertEqual(
                hashlib.sha256((SPLIT_DIR / name).read_bytes()).hexdigest(),
                expected_hash,
            )

        group_splits = {}
        for recording_id, mapping in group_map['recordings'].items():
            self.assertIn(recording_id, manifests[mapping['split']])
            previous = group_splits.setdefault(
                mapping['group_key'], mapping['split']
            )
            self.assertEqual(previous, mapping['split'])
        self.assertEqual(len(group_splits), 242)


if __name__ == '__main__':
    unittest.main()
