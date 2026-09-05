# SPDX-License-Identifier: Apache-2.0
# Includes modifications to ByteDance's piano_transcription implementation.

from __future__ import annotations

import logging
import os
import sys
import time

import torch
import torch.utils.data
from hydra import compose, initialize
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except Exception:
    wandb = None

from data_generator import (
    Augmentor,
    ChoralSATBDataset,
    EvalSampler,
    MAPS_Dataset,
    Maestro_Dataset,
    SMD_Dataset,
    BasePianoDataset,
    Sampler,
    collate_fn,
)
from evaluate import SegmentEvaluator
from losses import get_loss_func, resolve_loss_type
from models import build_model
from utilities import create_folder, create_logging, get_model_name, get_task_spec, move_data_to_device


DATASET_CLASS_MAP = {
    'maestro': Maestro_Dataset,
    'smd': SMD_Dataset,
    'maps': MAPS_Dataset,
    'cantoria': lambda cfg, is_training: BasePianoDataset(cfg, 'cantoria', is_training),
    'csd': lambda cfg, is_training: BasePianoDataset(cfg, 'csd', is_training),
    'youchorale': lambda cfg, is_training: BasePianoDataset(cfg, 'youchorale', is_training),
    'youchorale_pro': lambda cfg, is_training: BasePianoDataset(cfg, 'youchorale_pro', is_training),
}



def _build_dataset(cfg, dataset_name: str, is_training: bool):
    if getattr(cfg.choral, 'enable', False) and dataset_name in {'youchorale', 'youchorale_pro', 'csd', 'cantoria'}:
        return ChoralSATBDataset(cfg, dataset_name, is_training=is_training)
    dataset_builder = DATASET_CLASS_MAP[dataset_name]
    return dataset_builder(cfg, is_training=is_training)



def _build_optimizer(cfg, model):
    optim_name = getattr(cfg.exp, 'optim', 'adam').lower()
    if optim_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=cfg.exp.learning_rate)
    if optim_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=cfg.exp.learning_rate)
    raise ValueError(f'Unsupported optimizer: {cfg.exp.optim}')



def _wandb_init(cfg, run_id=None):
    if not getattr(cfg.wandb, 'enable', False):
        return None
    if wandb is None:
        logging.warning('wandb is enabled in config but not installed. Continue without wandb.')
        return None
    spec = get_task_spec(cfg)
    return wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        id=run_id,
        resume='must' if run_id else 'allow',
        config={
            'model_arch': spec.arch,
            'model_mode': spec.mode,
            'audio_feature': cfg.feature.audio_feature,
            'sample_rate': cfg.feature.sample_rate,
            'frames_per_second': cfg.feature.frames_per_second,
            'train_set': cfg.dataset.train_set,
        },
    )


def _tensorboard_init(cfg, model_name):
    if not getattr(cfg.tensorboard, 'enable', True):
        return None
    tb_dir = os.path.join(cfg.tensorboard.dir, model_name)
    create_folder(tb_dir)
    return SummaryWriter(log_dir=tb_dir)



def _prepare_batch(batch_data_dict, device):
    for key in batch_data_dict.keys():
        batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
    return batch_data_dict



def forward_pass(cfg, model, batch_data_dict, device):
    batch_data_dict = _prepare_batch(batch_data_dict, device)
    batch_output_dict = model(batch_data_dict['waveform'])
    loss_type = resolve_loss_type(cfg)
    loss = get_loss_func(loss_type)(model, batch_output_dict, batch_data_dict)
    return batch_output_dict, loss



def get_sampler(cfg, purpose, split, is_eval=None):
    sampler_cls = {'train': Sampler, 'eval': EvalSampler}[purpose]
    return sampler_cls(cfg, split=split, is_eval=is_eval)



def train(cfg):
    spec = get_task_spec(cfg)
    device = torch.device('cuda') if cfg.exp.cuda and torch.cuda.is_available() else torch.device('cpu')

    if getattr(cfg.feature, 'use_augmentation', False):
        cfg.feature.augmentor = Augmentor(cfg)
    else:
        cfg.feature.augmentor = None

    model = build_model(cfg).to(device)
    optimizer = _build_optimizer(cfg, model)

    model_name = get_model_name(cfg)
    checkpoints_dir = os.path.join(cfg.exp.workspace, 'checkpoints', model_name)
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', model_name)
    create_folder(checkpoints_dir)
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info(cfg)
    logging.info('Using device: %s', device)
    logging.info('Resolved task: arch=%s mode=%s', spec.arch, spec.mode)
    logging.info('Resolved post processor: %s', cfg.post.post_processor_type)

    start_iteration = 0
    wandb_run_id = None
    if cfg.exp.resume_iteration > 0:
        checkpoint_path = os.path.join(checkpoints_dir, f'{cfg.exp.resume_iteration}_iteration.pth')
        if os.path.exists(checkpoint_path):
            logging.info('Loading checkpoint %s', checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_iteration = int(checkpoint['iteration'])
            wandb_run_id = checkpoint.get('wandb_run_id')
        else:
            logging.warning('Checkpoint %s not found. Training starts from scratch.', checkpoint_path)

    wandb_run = _wandb_init(cfg, wandb_run_id)
    tb_writer = _tensorboard_init(cfg, model_name)

    train_dataset = _build_dataset(cfg, cfg.dataset.train_set, is_training=True)
    eval_train_dataset = _build_dataset(cfg, cfg.dataset.train_set, is_training=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=get_sampler(cfg, purpose='train', split='train'),
        collate_fn=collate_fn,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
    )
    eval_train_loader = torch.utils.data.DataLoader(
        dataset=eval_train_dataset,
        batch_sampler=get_sampler(cfg, purpose='eval', split='train'),
        collate_fn=collate_fn,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
    )

    eval_loaders = {}
    eval_split = 'validation'
    for dataset_name in list(dict.fromkeys([cfg.dataset.train_set, cfg.dataset.test_set])):
        eval_dataset = _build_dataset(cfg, dataset_name, is_training=False)
        eval_loaders[dataset_name] = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_sampler=get_sampler(cfg, purpose='eval', split=eval_split, is_eval=dataset_name),
            collate_fn=collate_fn,
            num_workers=cfg.exp.num_workers,
            pin_memory=True,
        )

    evaluator = SegmentEvaluator(model, cfg)
    iteration = start_iteration
    train_bgn_time = time.time()
    running_train_loss = 0.0
    loss_type = resolve_loss_type(cfg)
    logging.info('Resolved loss_type: %s', loss_type)

    optimizer.zero_grad(set_to_none=True)

    for batch_data_dict in train_loader:
        if cfg.exp.decay and iteration % cfg.exp.reduce_iteration == 0 and iteration != start_iteration and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        model.train()
        _, loss = forward_pass(cfg, model, batch_data_dict, device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running_train_loss += float(loss.item())

        should_log = (iteration == start_iteration) or ((iteration + 1) % cfg.exp.eval_iteration == 0)
        should_save = (iteration == start_iteration) or ((iteration + 1) % cfg.exp.save_iteration == 0)

        if should_log:
            train_fin_time = time.time()
            averaged_train_loss = running_train_loss / max(1, cfg.exp.eval_iteration)
            train_statistics = evaluator.evaluate(eval_train_loader)
            valid_statistics = {name: evaluator.evaluate(loader) for name, loader in eval_loaders.items()}

            logging.info('------------------------------------')
            logging.info('Iteration: %d / %d', iteration, cfg.exp.total_iteration)
            logging.info('Train loss: %.4f', averaged_train_loss)
            logging.info('Train statistics: %s', train_statistics)
            for name, stats in valid_statistics.items():
                logging.info('Eval %s statistics: %s', name, stats)

            if wandb_run is not None:
                log_dict = {'iteration': iteration, 'train_loss': averaged_train_loss}
                for key, value in train_statistics.items():
                    log_dict[f'train/{key}'] = value
                for dataset_name, stats in valid_statistics.items():
                    for key, value in stats.items():
                        log_dict[f'{dataset_name}/{key}'] = value
                wandb.log(log_dict)

            if tb_writer is not None:
                tb_writer.add_scalar('train/loss', averaged_train_loss, iteration)
                for key, value in train_statistics.items():
                    tb_writer.add_scalar(f'train/{key}', value, iteration)
                for dataset_name, stats in valid_statistics.items():
                    for key, value in stats.items():
                        tb_writer.add_scalar(f'{dataset_name}/{key}', value, iteration)
                tb_writer.flush()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            logging.info('Train time: %.3f s, validate time: %.3f s', train_time, validate_time)
            running_train_loss = 0.0
            train_bgn_time = time.time()

        if should_save:
            checkpoint = {
                'iteration': iteration,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'wandb_run_id': wandb_run.id if wandb_run is not None else None,
                'model_arch': spec.arch,
                'model_mode': spec.mode,
            }
            checkpoint_path = os.path.join(checkpoints_dir, f'{iteration}_iteration.pth')
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to %s', checkpoint_path)

        iteration += 1
        if iteration >= cfg.exp.total_iteration:
            break

    if wandb_run is not None:
        wandb.finish()
    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    initialize(config_path='./', job_name='train', version_base=None)
    cfg = compose(config_name='config', overrides=sys.argv[1:])
    train(cfg)
