# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

if not torch.cuda.is_available():
    torch.set_num_threads(1)

from feature_extractor import get_feature_extractor_and_bins
from utilities import get_task_spec


DEFAULT_MOMENTUM = 0.01


def init_layer(layer: nn.Module) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_gru(rnn: nn.GRU) -> None:
    for name, param in rnn.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)


def init_lstm(rnn: nn.LSTM) -> None:
    for name, param in rnn.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = DEFAULT_MOMENTUM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.init_weight()

    def init_weight(self) -> None:
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        return x


class AcousticModelCRnn8Dropout(nn.Module):
    def __init__(self, classes_num: int, freq_bins: int, momentum: float = DEFAULT_MOMENTUM):
        super().__init__()
        self.conv_block1 = ConvBlock(1, 48, momentum)
        self.conv_block2 = ConvBlock(48, 64, momentum)
        self.conv_block3 = ConvBlock(64, 96, momentum)
        self.conv_block4 = ConvBlock(96, 128, momentum)

        midfeat = self._infer_midfeat(freq_bins)
        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        self.gru = nn.GRU(768, 256, num_layers=2, bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        self.fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()

    def _infer_midfeat(self, freq_bins: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros((1, 1, 32, freq_bins), dtype=torch.float32)
            x = self.conv_block4(self.conv_block3(self.conv_block2(self.conv_block1(dummy))))
        return int(x.shape[1] * x.shape[3])

    def init_weight(self) -> None:
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x, _ = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return torch.sigmoid(self.fc(x))


class AcousticModelCRnn8Encoder(nn.Module):
    def __init__(self, freq_bins: int, momentum: float = DEFAULT_MOMENTUM):
        super().__init__()
        self.conv_block1 = ConvBlock(1, 48, momentum)
        self.conv_block2 = ConvBlock(48, 64, momentum)
        self.conv_block3 = ConvBlock(64, 96, momentum)
        self.conv_block4 = ConvBlock(96, 128, momentum)

        midfeat = self._infer_midfeat(freq_bins)
        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        self.gru = nn.GRU(768, 256, num_layers=2, bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        self.init_weight()

    def _infer_midfeat(self, freq_bins: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros((1, 1, 32, freq_bins), dtype=torch.float32)
            x = self.conv_block4(self.conv_block3(self.conv_block2(self.conv_block1(dummy))))
        return int(x.shape[1] * x.shape[3])

    def init_weight(self) -> None:
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x, _ = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x


class FeatureExtractorMixin:
    def _init_feature_extractor(self, cfg) -> None:
        self.cfg = cfg
        self.feature_extractor, self.freq_bins = get_feature_extractor_and_bins(
            cfg.feature.audio_feature,
            cfg.feature.sample_rate,
            cfg.feature.fft_size,
            cfg.feature.frames_per_second,
        )
        self.bn0 = nn.BatchNorm2d(self.freq_bins, momentum=DEFAULT_MOMENTUM)
        init_bn(self.bn0)

    def _waveform_to_feature_map(self, waveform: torch.Tensor) -> torch.Tensor:
        feature = self.feature_extractor(waveform)
        if isinstance(feature, tuple):
            feature = feature[0]
        if feature.ndim == 2:
            feature = feature.unsqueeze(1)
        if feature.ndim != 3:
            raise ValueError(f'Expected feature tensor with 3 dims, got {feature.shape}')
        feature = feature.unsqueeze(-1)
        feature = self.bn0(feature)
        feature = feature.transpose(1, 3)
        return feature

    def _waveform_to_feature_sequence(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._waveform_to_feature_map(waveform).squeeze(1)


class PedalCRNN(nn.Module, FeatureExtractorMixin):
    def __init__(self, cfg):
        super().__init__()
        self._init_feature_extractor(cfg)
        self.pedal_onset_model = AcousticModelCRnn8Dropout(1, self.freq_bins)
        self.pedal_offset_model = AcousticModelCRnn8Dropout(1, self.freq_bins)
        self.pedal_frame_model = AcousticModelCRnn8Dropout(1, self.freq_bins)

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._waveform_to_feature_map(waveform)
        return {
            'pedal_onset_output': self.pedal_onset_model(x),
            'pedal_offset_output': self.pedal_offset_model(x),
            'pedal_frame_output': self.pedal_frame_model(x),
        }


class PagCT(nn.Module, FeatureExtractorMixin):
    """Part-agnostic choral transcription model."""

    def __init__(self, cfg):
        super().__init__()
        self._init_feature_extractor(cfg)
        self.spec = get_task_spec(cfg)
        classes_num = cfg.feature.classes_num

        self.frame_model = AcousticModelCRnn8Dropout(classes_num, self.freq_bins)
        self.onset_model = AcousticModelCRnn8Dropout(classes_num, self.freq_bins)
        self.offset_model = AcousticModelCRnn8Dropout(classes_num, self.freq_bins) if self.spec.offset else None

        frame_inputs = 2 + int(self.spec.offset)
        self.frame_gru = nn.GRU(classes_num * frame_inputs, 256, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)
        init_gru(self.frame_gru)
        init_layer(self.frame_fc)

        self.pedal_model = PedalCRNN(cfg) if self.spec.pedal else None

    def _build_frame_input(
        self,
        frame_output: torch.Tensor,
        onset_output: torch.Tensor,
        offset_output: torch.Tensor | None,
    ) -> torch.Tensor:
        inputs: List[torch.Tensor] = [frame_output, onset_output.detach()]
        if offset_output is not None:
            inputs.append(offset_output.detach())
        return torch.cat(inputs, dim=2)

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._waveform_to_feature_map(waveform)
        frame_output = self.frame_model(x)
        onset_output = self.onset_model(x)
        offset_output = self.offset_model(x) if self.offset_model is not None else None

        frame_input = self._build_frame_input(frame_output, onset_output, offset_output)
        frame_input, _ = self.frame_gru(frame_input)
        frame_input = F.dropout(frame_input, p=0.5, training=self.training)
        frame_output = torch.sigmoid(self.frame_fc(frame_input))

        output_dict: Dict[str, torch.Tensor] = {
            'onset_output': onset_output,
            'frame_output': frame_output,
        }
        if offset_output is not None:
            output_dict['offset_output'] = offset_output
        if self.pedal_model is not None:
            output_dict.update(self.pedal_model(waveform))
        return output_dict


class PawCT(nn.Module, FeatureExtractorMixin):
    """Part-aware choral transcription model with SATB output heads."""

    def __init__(self, cfg):
        super().__init__()
        self._init_feature_extractor(cfg)
        self.cfg = cfg
        self.spec = get_task_spec(cfg)
        self.num_voices = int(getattr(cfg.choral, 'num_voices', 4))
        self.use_presence_head = bool(getattr(cfg.choral, 'use_presence_head', True))
        # Historical checkpoints used the clip-level presence probability as a
        # hard multiplicative cap on every note probability.  New configs keep
        # the head as an auxiliary task without suppressing note recall.
        self.apply_presence_gate = bool(getattr(cfg.choral, 'apply_presence_gate', True))
        self.assignment_module_name = str(getattr(cfg.choral, 'assignment_module', 'heads')).strip()
        self.assignment_temperature = float(getattr(cfg.choral, 'assignment_temperature', 1.0))
        self.voice_interaction_module_name = str(getattr(cfg.choral, 'voice_interaction_module', 'none')).strip()
        classes_num = cfg.feature.classes_num

        self.shared_encoder = AcousticModelCRnn8Encoder(self.freq_bins)
        hidden_size = 512

        if self.assignment_module_name == 'heads':
            self.onset_heads = nn.ModuleList([nn.Linear(hidden_size, classes_num) for _ in range(self.num_voices)])
            self.frame_seed_heads = nn.ModuleList([nn.Linear(hidden_size, classes_num) for _ in range(self.num_voices)])
            self.offset_heads = nn.ModuleList([nn.Linear(hidden_size, classes_num) for _ in range(self.num_voices)]) if self.spec.offset else None
            self.onset_head = None
            self.frame_seed_head = None
            self.offset_head = None
            self.assignment_net = None
            for head in list(self.onset_heads) + list(self.frame_seed_heads):
                init_layer(head)
            if self.offset_heads is not None:
                for head in self.offset_heads:
                    init_layer(head)
        else:
            self.onset_heads = None
            self.frame_seed_heads = None
            self.offset_heads = None
            self.onset_head = nn.Linear(hidden_size, classes_num)
            self.frame_seed_head = nn.Linear(hidden_size, classes_num)
            self.offset_head = nn.Linear(hidden_size, classes_num) if self.spec.offset else None
            init_layer(self.onset_head)
            init_layer(self.frame_seed_head)
            if self.offset_head is not None:
                init_layer(self.offset_head)
            assignment_in_channels = 2 + int(self.spec.offset)
            assignment_hidden = int(getattr(cfg.choral, 'assignment_hidden_channels', 64))
            if self.assignment_module_name == 'va2_cnn':
                self.assignment_net = VA2CNNAssignment(assignment_in_channels, self.num_voices, assignment_hidden)
            elif self.assignment_module_name == 'va2_convlstm':
                assignment_rnn_hidden = int(getattr(cfg.choral, 'assignment_rnn_hidden_size', 128))
                self.assignment_net = VA2ConvLSTMAssignment(
                    assignment_in_channels,
                    classes_num,
                    self.num_voices,
                    assignment_hidden,
                    assignment_rnn_hidden,
                )
            else:
                raise ValueError(f'Unsupported choral.assignment_module={self.assignment_module_name}')

        voice_token_dim = classes_num * (2 + int(self.spec.offset))
        if self.voice_interaction_module_name == 'none':
            self.voice_interaction_block = None
        elif self.voice_interaction_module_name == 'self_attn':
            interaction_dim = int(getattr(cfg.choral, 'voice_interaction_dim', 256))
            interaction_heads = int(getattr(cfg.choral, 'voice_interaction_heads', 4))
            interaction_layers = int(getattr(cfg.choral, 'voice_interaction_layers', 1))
            interaction_dropout = float(getattr(cfg.choral, 'voice_interaction_dropout', 0.1))
            self.voice_interaction_block = CrossVoiceSelfAttentionBlock(
                token_dim=voice_token_dim,
                num_voices=self.num_voices,
                interaction_dim=interaction_dim,
                num_heads=interaction_heads,
                num_layers=interaction_layers,
                dropout=interaction_dropout,
            )
        else:
            raise ValueError(f'Unsupported choral.voice_interaction_module={self.voice_interaction_module_name}')

        combined_inputs = self.num_voices * classes_num * (2 + int(self.spec.offset))
        self.frame_gru = nn.GRU(combined_inputs, 256, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        self.frame_fc = nn.Linear(512, self.num_voices * classes_num, bias=True)
        init_gru(self.frame_gru)
        init_layer(self.frame_fc)

        if self.use_presence_head:
            self.presence_fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, self.num_voices),
            )
            init_layer(self.presence_fc[0])
            init_layer(self.presence_fc[2])
        else:
            self.presence_fc = None

    def _apply_voice_heads(self, hidden: torch.Tensor, heads: nn.ModuleList) -> torch.Tensor:
        outputs = [torch.sigmoid(head(hidden)) for head in heads]
        return torch.stack(outputs, dim=2)

    def _apply_shared_head(self, hidden: torch.Tensor, head: nn.Linear | None) -> torch.Tensor | None:
        if head is None:
            return None
        return torch.sigmoid(head(hidden))

    def _assign_shared_salience(
        self,
        frame_seed: torch.Tensor,
        onset_output: torch.Tensor,
        offset_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        assignment_inputs: List[torch.Tensor] = [frame_seed, onset_output]
        if offset_output is not None:
            assignment_inputs.append(offset_output)
        assignment_stack = torch.stack(assignment_inputs, dim=1)
        assignment_logits = self.assignment_net(assignment_stack)
        assignment_weights = torch.softmax(assignment_logits / max(self.assignment_temperature, 1e-6), dim=2)

        voice_frame_seed = frame_seed.unsqueeze(2) * assignment_weights
        voice_onset_output = onset_output.unsqueeze(2) * assignment_weights
        voice_offset_output = offset_output.unsqueeze(2) * assignment_weights if offset_output is not None else None
        return voice_frame_seed, voice_onset_output, voice_offset_output, assignment_logits, assignment_weights

    def _presence_gate(self, hidden: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.presence_fc is None:
            return None, None
        pooled = hidden.mean(dim=1)
        logits = self.presence_fc(pooled)
        gate = torch.sigmoid(logits)[:, None, :, None]
        return logits, gate

    def _apply_voice_interaction(
        self,
        voice_frame_seed: torch.Tensor,
        voice_onset_output: torch.Tensor,
        voice_offset_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.voice_interaction_block is None:
            return voice_frame_seed, voice_onset_output, voice_offset_output

        streams: List[torch.Tensor] = [voice_frame_seed, voice_onset_output]
        if voice_offset_output is not None:
            streams.append(voice_offset_output)

        stream_logits = [torch.logit(stream.clamp(1e-4, 1.0 - 1e-4)) for stream in streams]
        combined_logits = torch.cat(stream_logits, dim=3)
        refined_logits = self.voice_interaction_block(combined_logits)
        split_sizes = [self.cfg.feature.classes_num for _ in streams]
        refined_streams = torch.split(refined_logits, split_sizes, dim=3)
        refined_probs = [torch.sigmoid(refined_stream) for refined_stream in refined_streams]

        refined_frame_seed = refined_probs[0]
        refined_onset_output = refined_probs[1]
        refined_offset_output = refined_probs[2] if voice_offset_output is not None else None
        return refined_frame_seed, refined_onset_output, refined_offset_output

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._waveform_to_feature_map(waveform)
        hidden = self.shared_encoder(x)

        assignment_logits = None
        assignment_weights = None
        if self.assignment_module_name == 'heads':
            voice_onset_output = self._apply_voice_heads(hidden, self.onset_heads)
            voice_frame_seed = self._apply_voice_heads(hidden, self.frame_seed_heads)
            voice_offset_output = self._apply_voice_heads(hidden, self.offset_heads) if self.offset_heads is not None else None
        else:
            onset_output = self._apply_shared_head(hidden, self.onset_head)
            frame_seed = self._apply_shared_head(hidden, self.frame_seed_head)
            offset_output = self._apply_shared_head(hidden, self.offset_head)
            voice_frame_seed, voice_onset_output, voice_offset_output, assignment_logits, assignment_weights = self._assign_shared_salience(
                frame_seed,
                onset_output,
                offset_output,
            )

        voice_frame_seed, voice_onset_output, voice_offset_output = self._apply_voice_interaction(
            voice_frame_seed,
            voice_onset_output,
            voice_offset_output,
        )

        frame_inputs: List[torch.Tensor] = [voice_frame_seed.flatten(2), voice_onset_output.detach().flatten(2)]
        if voice_offset_output is not None:
            frame_inputs.append(voice_offset_output.detach().flatten(2))
        frame_hidden, _ = self.frame_gru(torch.cat(frame_inputs, dim=2))
        frame_hidden = F.dropout(frame_hidden, p=0.5, training=self.training)
        voice_frame_output = torch.sigmoid(self.frame_fc(frame_hidden)).reshape(
            frame_hidden.shape[0], frame_hidden.shape[1], self.num_voices, self.cfg.feature.classes_num
        )

        voice_presence_logits, presence_gate = self._presence_gate(hidden)
        if presence_gate is not None and self.apply_presence_gate:
            voice_onset_output = voice_onset_output * presence_gate
            voice_frame_output = voice_frame_output * presence_gate
            if voice_offset_output is not None:
                voice_offset_output = voice_offset_output * presence_gate

        output_dict: Dict[str, torch.Tensor] = {
            'voice_onset_output': voice_onset_output,
            'voice_frame_output': voice_frame_output,
            'onset_output': torch.max(voice_onset_output, dim=2).values,
            'frame_output': torch.max(voice_frame_output, dim=2).values,
        }
        if voice_offset_output is not None:
            output_dict['voice_offset_output'] = voice_offset_output
            output_dict['offset_output'] = torch.max(voice_offset_output, dim=2).values
        if voice_presence_logits is not None:
            output_dict['voice_presence_logits'] = voice_presence_logits
            output_dict['voice_presence_output'] = torch.sigmoid(voice_presence_logits)
        if assignment_logits is not None and assignment_weights is not None:
            output_dict['voice_assignment_logits'] = assignment_logits
            output_dict['voice_assignment_output'] = assignment_weights
        return output_dict


class VA2CNNAssignment(nn.Module):
    def __init__(self, in_channels: int, num_voices: int, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, momentum=DEFAULT_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, momentum=DEFAULT_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, num_voices, kernel_size=1, bias=True),
        )
        init_layer(self.net[0])
        init_bn(self.net[1])
        init_layer(self.net[3])
        init_bn(self.net[4])
        init_layer(self.net[6])

    def forward(self, salience_stack: torch.Tensor) -> torch.Tensor:
        logits = self.net(salience_stack)
        return logits.permute(0, 2, 1, 3)


class CrossVoiceSelfAttentionLayer(nn.Module):
    def __init__(self, interaction_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(interaction_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=interaction_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(interaction_dim)
        self.ffn = nn.Sequential(
            nn.Linear(interaction_dim, interaction_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_dim * 4, interaction_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        init_layer(self.ffn[0])
        init_layer(self.ffn[3])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_inputs = self.norm1(tokens)
        attn_output, _ = self.attn(attn_inputs, attn_inputs, attn_inputs, need_weights=False)
        tokens = tokens + self.dropout1(attn_output)
        tokens = tokens + self.dropout2(self.ffn(self.norm2(tokens)))
        return tokens


class CrossVoiceSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_voices: int,
        interaction_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.num_voices = int(num_voices)
        self.token_dim = int(token_dim)
        self.project_in = nn.Linear(token_dim, interaction_dim)
        self.voice_embedding = nn.Embedding(num_voices, interaction_dim)
        self.layers = nn.ModuleList(
            [CrossVoiceSelfAttentionLayer(interaction_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.project_out = nn.Linear(interaction_dim, token_dim)
        init_layer(self.project_in)
        init_layer(self.project_out)
        nn.init.normal_(self.voice_embedding.weight, mean=0.0, std=0.02)

    def forward(self, voice_logits: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_voices, token_dim = voice_logits.shape
        if num_voices != self.num_voices or token_dim != self.token_dim:
            raise ValueError(
                f'CrossVoiceSelfAttentionBlock expected (*, *, {self.num_voices}, {self.token_dim}), '
                f'but received {tuple(voice_logits.shape)}'
            )

        residual = voice_logits.reshape(batch_size * time_steps, num_voices, token_dim)
        tokens = self.project_in(residual)
        voice_ids = torch.arange(num_voices, device=voice_logits.device)
        tokens = tokens + self.voice_embedding(voice_ids)[None, :, :]
        for layer in self.layers:
            tokens = layer(tokens)
        delta = self.project_out(tokens)
        refined = residual + delta
        return refined.reshape(batch_size, time_steps, num_voices, token_dim)


class VA2ConvLSTMAssignment(nn.Module):
    def __init__(
        self,
        in_channels: int,
        classes_num: int,
        num_voices: int,
        hidden_channels: int,
        rnn_hidden_size: int,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, momentum=DEFAULT_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, momentum=DEFAULT_MOMENTUM),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_size=hidden_channels * classes_num,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(rnn_hidden_size * 2, num_voices * classes_num)
        init_layer(self.conv[0])
        init_bn(self.conv[1])
        init_layer(self.conv[3])
        init_bn(self.conv[4])
        init_lstm(self.rnn)
        init_layer(self.fc)
        self.num_voices = num_voices
        self.classes_num = classes_num

    def forward(self, salience_stack: torch.Tensor) -> torch.Tensor:
        x = self.conv(salience_stack)
        x = x.permute(0, 2, 1, 3).flatten(2)
        x, _ = self.rnn(x)
        logits = self.fc(x)
        return logits.reshape(x.shape[0], x.shape[1], self.num_voices, self.classes_num)


class OnfConvStack(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, output_features // 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            nn.Conv2d(output_features // 16, output_features // 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5),
        )

    def forward(self, feature_seq: torch.Tensor) -> torch.Tensor:
        x = feature_seq.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        return self.fc(x)


class BiLSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.rnn(x)
        return x


class OnsetBranch(nn.Module):
    def __init__(self, input_features: int, output_features: int, model_size: int):
        super().__init__()
        self.net = nn.Sequential(
            OnfConvStack(input_features, model_size),
            BiLSTMBlock(model_size, model_size // 2),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleBranch(nn.Module):
    def __init__(self, input_features: int, output_features: int, model_size: int):
        super().__init__()
        self.net = nn.Sequential(
            OnfConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlexibleOnsetsAndFrames(nn.Module, FeatureExtractorMixin):
    def __init__(self, cfg):
        super().__init__()
        self._init_feature_extractor(cfg)
        self.spec = get_task_spec(cfg)
        if self.spec.mode not in {'frame_onset', 'frame_onset_offset'}:
            raise ValueError(f'FlexibleOnsetsAndFrames does not support mode={self.spec.mode}')

        output_features = cfg.feature.classes_num
        model_size = 48 * 16

        self.onset_stack = OnsetBranch(self.freq_bins, output_features, model_size)
        self.frame_stack = SimpleBranch(self.freq_bins, output_features, model_size)
        self.offset_stack = OnsetBranch(self.freq_bins, output_features, model_size) if self.spec.offset else None

        combined_inputs = output_features * (2 + int(self.spec.offset))
        self.combined_stack = nn.Sequential(
            BiLSTMBlock(combined_inputs, model_size // 2),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_seq = self._waveform_to_feature_sequence(waveform)
        onset_output = self.onset_stack(feat_seq)
        activation_output = self.frame_stack(feat_seq)
        offset_output = self.offset_stack(feat_seq) if self.offset_stack is not None else None

        combined_inputs = [onset_output.detach(), activation_output]
        if offset_output is not None:
            combined_inputs.insert(1, offset_output.detach())
        frame_output = self.combined_stack(torch.cat(combined_inputs, dim=-1))

        output_dict: Dict[str, torch.Tensor] = {
            'onset_output': onset_output,
            'frame_output': frame_output,
        }
        if offset_output is not None:
            output_dict['offset_output'] = offset_output
        return output_dict


# Backward-compatible API names for historical configs, imports, and serialized
# model objects. New code should import ``PagCT`` and ``PawCT`` directly. The
# aliases deliberately point to the same classes, so state-dict keys are
# unchanged and historical checkpoints remain loadable.
FlexibleHPT = PagCT
FlexibleHPTChoralStream = PawCT


def build_model(cfg) -> nn.Module:
    spec = get_task_spec(cfg)
    choral_enabled = bool(getattr(cfg.choral, 'enable', False))

    if spec.arch == 'pawct':
        if not choral_enabled:
            raise ValueError("model.arch='pawct' requires choral.enable=true")
        return PawCT(cfg)
    if spec.arch == 'pagct':
        if choral_enabled:
            raise ValueError("model.arch='pagct' requires choral.enable=false")
        return PagCT(cfg)

    # ``hpt`` is the historical architecture selector. Keep this branch so old
    # resolved configurations select the same model family without changing
    # checkpoint parameter names.
    if spec.arch == 'hpt':
        return PawCT(cfg) if choral_enabled else PagCT(cfg)
    if spec.arch == 'onf':
        if choral_enabled:
            raise ValueError("model.arch='onf' requires choral.enable=false")
        return FlexibleOnsetsAndFrames(cfg)
    raise ValueError(f'Unsupported model.arch={spec.arch}')
