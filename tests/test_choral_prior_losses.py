import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "src"))

from losses import (
    bce,
    choral_bce,
    choral_task_bce,
    voice_continuity_prior_loss,
    voice_range_prior_loss,
)


class ChoralPriorLossTest(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(
            feature=SimpleNamespace(begin_note=21, classes_num=88),
            choral=SimpleNamespace(
                voice_assignment_range_mins=[60, 55, 48, 40],
                voice_assignment_range_maxs=[88, 79, 72, 67],
                voice_assignment_range_margin=0.0,
            ),
        )

    def test_positive_weight_increases_rare_positive_gradient(self):
        output = torch.full((1, 2), 0.1, requires_grad=True)
        target = torch.tensor([[1.0, 0.0]])
        loss = choral_bce(output, target, positive_weight=10.0)
        loss.backward()
        self.assertGreater(abs(float(output.grad[0, 0])), abs(float(output.grad[0, 1])))

    def test_bce_is_finite_for_saturated_half_precision_probabilities(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                output = torch.tensor([0.0, 1.0], dtype=dtype, requires_grad=True)
                target = torch.tensor([1.0, 0.0], dtype=dtype)
                loss = bce(output, target)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(output.grad).all())

    def test_range_prior_penalizes_out_of_range_more(self):
        in_range = torch.zeros((1, 1, 4, 88))
        out_of_range = torch.zeros_like(in_range)
        in_range[0, 0, 0, 72 - 21] = 1.0
        out_of_range[0, 0, 0, 40 - 21] = 1.0
        self.assertEqual(float(voice_range_prior_loss(in_range, self.cfg)), 0.0)
        self.assertGreater(float(voice_range_prior_loss(out_of_range, self.cfg)), 0.0)

    def test_range_prior_gives_confident_violations_stronger_logit_gradients(self):
        logits = torch.full((1, 1, 4, 88), -20.0, requires_grad=True)
        with torch.no_grad():
            # Both notes are below the soprano range. A 99% violation should
            # receive more correction than a 50% violation, not less.
            logits[0, 0, 0, 40 - 21] = 0.0
            logits[0, 0, 0, 41 - 21] = torch.logit(torch.tensor(0.99))

        loss = voice_range_prior_loss(torch.sigmoid(logits), self.cfg)
        loss.backward()

        moderate_gradient = abs(float(logits.grad[0, 0, 0, 40 - 21]))
        confident_gradient = abs(float(logits.grad[0, 0, 0, 41 - 21]))
        self.assertGreater(confident_gradient, moderate_gradient)

    def test_range_prior_never_penalizes_a_trusted_out_of_range_positive(self):
        output = torch.zeros((1, 1, 4, 88))
        output[0, 0, 0, 40 - 21] = 1.0
        target = output.clone()
        mask = torch.ones_like(output)

        self.assertEqual(
            float(voice_range_prior_loss(output, self.cfg, target=target, mask=mask)),
            0.0,
        )

    def test_range_prior_has_no_gradient_on_a_trusted_out_of_range_positive(self):
        output = torch.zeros((1, 1, 4, 88), requires_grad=True)
        target = torch.zeros_like(output)
        with torch.no_grad():
            output[0, 0, 0, 40 - 21] = 0.9
            target[0, 0, 0, 40 - 21] = 1.0

        loss = voice_range_prior_loss(
            output,
            self.cfg,
            target=target,
            mask=torch.ones_like(output),
        )
        loss.backward()

        self.assertEqual(float(output.grad[0, 0, 0, 40 - 21]), 0.0)

    def test_range_prior_respects_supervision_mask(self):
        output = torch.zeros((1, 1, 4, 88))
        output[0, 0, 0, 40 - 21] = 1.0
        mask = torch.ones_like(output)
        mask[0, 0, 0, 40 - 21] = 0.0

        self.assertEqual(
            float(voice_range_prior_loss(output, self.cfg, mask=mask)),
            0.0,
        )

    def test_continuity_prior_penalizes_pitch_jump(self):
        smooth = torch.zeros((1, 2, 4, 88))
        jump = torch.zeros_like(smooth)
        target = torch.zeros_like(smooth)
        smooth[0, :, 0, 50] = 1.0
        jump[0, 0, 0, 20] = 1.0
        jump[0, 1, 0, 50] = 1.0
        target[0, :, 0, 50] = 1.0
        self.assertEqual(float(voice_continuity_prior_loss(smooth, target)), 0.0)
        self.assertGreater(float(voice_continuity_prior_loss(jump, target)), 0.0)

    def test_continuity_prior_does_not_flatten_a_true_melodic_leap(self):
        target = torch.zeros((1, 2, 1, 88))
        target[0, 0, 0, 20] = 1.0
        target[0, 1, 0, 50] = 1.0

        matching_prediction = target.clone()
        flattened_prediction = torch.zeros_like(target)
        flattened_prediction[0, :, 0, 20] = 1.0

        self.assertEqual(
            float(voice_continuity_prior_loss(matching_prediction, target)),
            0.0,
        )
        self.assertGreater(
            float(voice_continuity_prior_loss(flattened_prediction, target)),
            0.0,
        )

    def test_continuity_prior_treats_multi_pitch_onset_as_a_trajectory_break(self):
        target = torch.zeros((1, 3, 1, 88))
        target[0, 0, 0, 20] = 1.0
        target[0, 1, 0, 10] = 1.0
        target[0, 1, 0, 30] = 1.0
        target[0, 2, 0, 50] = 1.0

        # If the divisi frame were reduced to its centroid (20), the final
        # flattened prediction would incur a large, misleading OC penalty.
        prediction = torch.zeros_like(target, requires_grad=True)
        with torch.no_grad():
            prediction[0, :, 0, 20] = 1.0

        loss = voice_continuity_prior_loss(prediction, target)
        loss.backward()

        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction.grad)))

    def test_continuity_prior_ignores_masked_frames_and_gradients(self):
        output = torch.zeros((1, 2, 1, 4), requires_grad=True)
        with torch.no_grad():
            output[0, 0, 0, 0] = 1.0
            output[0, 1, 0, 3] = 1.0
        target = torch.zeros_like(output)
        target[0, :, 0, 0] = 1.0
        mask = torch.ones_like(output)
        mask[:, 1, :, :] = 0.0

        loss = voice_continuity_prior_loss(output, target, mask)
        loss.backward()

        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.equal(output.grad[:, 1], torch.zeros_like(output.grad[:, 1])))

    def test_continuity_prior_links_events_across_a_rest_without_length_dilution(self):
        adjacent_target = torch.zeros((1, 2, 1, 88))
        adjacent_target[0, 0, 0, 20] = 1.0
        adjacent_target[0, 1, 0, 32] = 1.0
        adjacent_flat = torch.zeros_like(adjacent_target)
        adjacent_flat[0, :, 0, 20] = 1.0

        separated_target = torch.zeros((1, 201, 1, 88))
        separated_target[0, 0, 0, 20] = 1.0
        separated_target[0, 200, 0, 32] = 1.0
        separated_flat = torch.zeros_like(separated_target)
        separated_flat[0, 0, 0, 20] = 1.0
        separated_flat[0, 200, 0, 20] = 1.0

        adjacent_loss = voice_continuity_prior_loss(adjacent_flat, adjacent_target)
        separated_loss = voice_continuity_prior_loss(
            separated_flat,
            separated_target,
            frames_per_second=100.0,
            gap_decay_seconds=0.0,
        )

        self.assertGreater(float(separated_loss), 0.1)
        self.assertAlmostEqual(float(separated_loss), float(adjacent_loss), places=6)

    def test_continuity_prior_uses_true_silent_gap_not_inter_onset_interval(self):
        target = torch.zeros((1, 101, 1, 88))
        target[0, 0, 0, 20] = 1.0
        target[0, 100, 0, 32] = 1.0
        flattened = torch.zeros_like(target)
        flattened[0, 0, 0, 20] = 1.0
        flattened[0, 100, 0, 20] = 1.0

        legato_activity = torch.zeros_like(target)
        legato_activity[0, :101, 0, 20] = 1.0
        rest_activity = torch.zeros_like(target)
        rest_activity[0, 0, 0, 20] = 1.0
        rest_activity[0, 100, 0, 32] = 1.0

        legato_loss = voice_continuity_prior_loss(
            flattened,
            target,
            activity_target=legato_activity,
            frames_per_second=100.0,
            gap_decay_seconds=1.0,
        )
        rest_loss = voice_continuity_prior_loss(
            flattened,
            target,
            activity_target=rest_activity,
            frames_per_second=100.0,
            gap_decay_seconds=1.0,
        )

        self.assertAlmostEqual(float(legato_loss), 0.5, places=6)
        self.assertGreater(float(rest_loss), 0.0)
        self.assertLess(float(rest_loss), float(legato_loss))

    def test_continuity_prior_is_finite_for_tiny_gap_decay(self):
        target = torch.zeros((1, 101, 1, 88))
        target[0, 0, 0, 20] = 1.0
        target[0, 100, 0, 32] = 1.0
        flattened = torch.zeros_like(target)
        flattened[0, 0, 0, 20] = 1.0
        flattened[0, 100, 0, 20] = 1.0

        loss = voice_continuity_prior_loss(
            flattened,
            target,
            frames_per_second=100.0,
            gap_decay_seconds=1e-300,
        )

        self.assertTrue(torch.isfinite(loss))

    def test_continuity_prior_tiny_decay_is_finite_for_zero_gap_and_gradients(self):
        target = torch.zeros((1, 2, 1, 88))
        target[0, 0, 0, 20] = 1.0
        target[0, 1, 0, 32] = 1.0
        activity = torch.zeros_like(target)
        activity[0, 0, 0, 20] = 1.0
        activity[0, 1, 0, 32] = 1.0
        flattened = torch.zeros_like(target, requires_grad=True)
        with torch.no_grad():
            flattened[0, :, 0, 20] = 1.0

        loss = voice_continuity_prior_loss(
            flattened,
            target,
            activity_target=activity,
            frames_per_second=100.0,
            gap_decay_seconds=1e-300,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss), 0.5, places=6)
        self.assertTrue(torch.isfinite(flattened.grad).all())

    def test_continuity_prior_rejects_nonfinite_time_parameters(self):
        values = torch.zeros((1, 2, 1, 4))
        for kwargs in (
            {'frames_per_second': float('inf')},
            {'frames_per_second': float('nan')},
            {'gap_decay_seconds': float('inf')},
            {'gap_decay_seconds': float('nan')},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                voice_continuity_prior_loss(values, values, **kwargs)

    def test_continuity_prior_excludes_pitch_selectively_masked_events(self):
        output = torch.zeros((1, 2, 1, 4))
        output[0, :, 0, 0] = 1.0
        output[0, :, 0, 3] = 1.0
        target = torch.zeros_like(output)
        target[0, :, 0, 0] = 1.0
        mask = torch.ones_like(output)
        mask[0, 1, 0, 3] = 0.0

        self.assertEqual(
            float(voice_continuity_prior_loss(output, target, mask)),
            0.0,
        )

    def test_choral_task_uses_onset_events_for_continuity(self):
        cfg = SimpleNamespace(
            model=SimpleNamespace(arch='pawct', mode='frame_onset'),
            feature=SimpleNamespace(frames_per_second=100.0),
            choral=SimpleNamespace(
                voice_onset_loss_weight=1.0,
                voice_onset_positive_weight=1.0,
                continuity_prior_loss_weight=0.0,
                oc_gap_decay_seconds=2.0,
            ),
        )
        model = SimpleNamespace(cfg=cfg)
        target = torch.zeros((1, 101, 1, 88))
        target[0, 0, 0, 20] = 1.0
        target[0, 100, 0, 32] = 1.0
        prediction = torch.full_like(target, 1e-4)
        prediction[0, 0, 0, 20] = 1.0 - 1e-4
        prediction[0, 100, 0, 20] = 1.0 - 1e-4
        output_dict = {'voice_onset_output': prediction}
        target_dict = {
            'voice_onset_roll': target,
            'voice_onset_mask_roll': torch.ones_like(target),
            'voice_frame_roll': target.clone(),
            'voice_frame_mask_roll': torch.ones_like(target),
        }

        without_oc = choral_task_bce(model, output_dict, target_dict)
        cfg.choral.continuity_prior_loss_weight = 1.0
        with_oc = choral_task_bce(model, output_dict, target_dict)

        self.assertGreater(float(with_oc), float(without_oc) + 0.1)

    def test_choral_task_requires_frame_activity_for_v3_continuity(self):
        cfg = SimpleNamespace(
            model=SimpleNamespace(arch='pawct', mode='frame_onset'),
            feature=SimpleNamespace(frames_per_second=100.0),
            choral=SimpleNamespace(
                voice_onset_loss_weight=1.0,
                voice_onset_positive_weight=1.0,
                continuity_prior_loss_weight=0.01,
                oc_gap_decay_seconds=2.0,
            ),
        )
        values = torch.zeros((1, 2, 1, 4))

        with self.assertRaisesRegex(ValueError, 'frame activity targets'):
            choral_task_bce(
                SimpleNamespace(cfg=cfg),
                {'voice_onset_output': values},
                {
                    'voice_onset_roll': values,
                    'voice_onset_mask_roll': torch.ones_like(values),
                },
            )


if __name__ == "__main__":
    unittest.main()
