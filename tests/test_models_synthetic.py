import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F


REPO_DIR = Path(__file__).resolve().parents[1]
for source_dir in (REPO_DIR / "src", REPO_DIR / "tools", REPO_DIR / "experiments"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import losses
import models


class DummyFeatureExtractor(torch.nn.Module):
    """Small deterministic front end that keeps the model test data-free."""

    def forward(self, waveform):
        frames = F.interpolate(waveform[:, None, :], size=24, mode="linear", align_corners=False).squeeze(1)
        scales = torch.linspace(0.5, 1.5, 16, device=waveform.device, dtype=waveform.dtype)
        return frames[:, None, :] * scales[None, :, None]


def make_cfg():
    return SimpleNamespace(
        feature=SimpleNamespace(
            audio_feature="logmel",
            sample_rate=16000,
            fft_size=2048,
            frames_per_second=100,
            classes_num=12,
            begin_note=48,
        ),
        model=SimpleNamespace(arch="pawct", mode="frame_onset_offset", type=None),
        choral=SimpleNamespace(
            enable=True,
            num_voices=4,
            use_presence_head=True,
            assignment_module="heads",
            voice_interaction_module="none",
            voice_frame_loss_weight=1.0,
            voice_onset_loss_weight=1.0,
            voice_offset_loss_weight=1.0,
            union_frame_loss_weight=1.0,
            union_onset_loss_weight=1.0,
            union_offset_loss_weight=1.0,
            presence_loss_weight=0.2,
        ),
    )


class PawCTSyntheticTest(unittest.TestCase):
    def test_forward_union_shapes_and_backward(self):
        cfg = make_cfg()
        with mock.patch.object(
            models,
            "get_feature_extractor_and_bins",
            return_value=(DummyFeatureExtractor(), 16),
        ):
            model = models.PawCT(cfg)

        waveform = torch.linspace(-1.0, 1.0, 320).reshape(2, 160)
        output = model(waveform)

        self.assertEqual(output["voice_frame_output"].shape, (2, 24, 4, 12))
        self.assertEqual(output["voice_onset_output"].shape, (2, 24, 4, 12))
        self.assertEqual(output["voice_offset_output"].shape, (2, 24, 4, 12))
        self.assertEqual(output["voice_presence_output"].shape, (2, 4))
        self.assertTrue(torch.isfinite(output["voice_frame_output"]).all())
        self.assertTrue(
            torch.allclose(
                output["frame_output"],
                output["voice_frame_output"].max(dim=2).values,
            )
        )

        targets = {
            "voice_frame_roll": torch.zeros_like(output["voice_frame_output"]),
            "voice_onset_roll": torch.zeros_like(output["voice_onset_output"]),
            "voice_offset_roll": torch.zeros_like(output["voice_offset_output"]),
            "voice_frame_mask_roll": torch.ones_like(output["voice_frame_output"]),
            "voice_onset_mask_roll": torch.ones_like(output["voice_onset_output"]),
            "voice_offset_mask_roll": torch.ones_like(output["voice_offset_output"]),
            "frame_roll": torch.zeros_like(output["frame_output"]),
            "onset_roll": torch.zeros_like(output["onset_output"]),
            "offset_roll": torch.zeros_like(output["offset_output"]),
            "frame_mask_roll": torch.ones_like(output["frame_output"]),
            "onset_mask_roll": torch.ones_like(output["onset_output"]),
            "offset_mask_roll": torch.ones_like(output["offset_output"]),
            "voice_presence": torch.zeros_like(output["voice_presence_output"]),
        }
        loss = losses.choral_task_bce(model, output, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
