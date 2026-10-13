"""Train LDA with WAM-TTT's human-inner / robot-outer meta objective."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.distributed.optim import ZeroRedundancyOptimizer

from lda.dataloader import collate_fn
from lda.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
from lda.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from lda.dataloader.task_batch_sampler import DistributedTaskBatchSampler
from lda.model.framework import build_framework
from lda.training.trainer_utils.trainer_tools import build_param_lr_groups, normalize_dotlist_args
from lda.training.wam_ttt import WAMTTTMetaLearner, phase_align_human_examples


def build_task_loaders(cfg):
    """Build separate video-only inner and robot-task outer loaders."""
    data_cfg = cfg.datasets.vla_data
    entries = cfg.datasets.wam_ttt.mixture
    data_root = Path(data_cfg.data_root_dir).expanduser()
    if not data_root.is_dir():
        raise FileNotFoundError(f"LeRobot data root does not exist: {data_root}")
    for item in entries:
        dataset_path = data_root / item.name
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Configured LeRobot dataset does not exist: {dataset_path}")
    data_mixture = [
        (
            make_LeRobotSingleDataset(
                data_root,
                item.name,
                item.robot_type,
                delete_pause_frame=bool(data_cfg.get("delete_pause_frame", False)),
                data_cfg=data_cfg,
                CoT_prompt=data_cfg.get("CoT_prompt"),
            ),
            float(item.weight),
        )
        for item in entries
    ]
    dataset = LeRobotMixtureDataset(
        data_mixture=data_mixture,
        mode="train",
        balance_dataset_weights=bool(cfg.datasets.wam_ttt.balance_dataset_weights),
        balance_trajectory_weights=bool(cfg.datasets.wam_ttt.balance_trajectory_weights),
        seed=int(cfg.seed),
        use_delta_action=bool(data_cfg.get("use_delta_action", False)),
        data_cfg=data_cfg,
        state_dim=cfg.framework.action_model.get("state_dim"),
        action_dim=int(cfg.framework.action_model.action_dim),
    )
    required_tasks = {"video_gen", *cfg.datasets.wam_ttt.outer_tasks}
    missing_tasks = required_tasks - set(dataset.task_to_dataset_indices)
    if missing_tasks:
        raise ValueError(f"Configured datasets do not support required tasks: {sorted(missing_tasks)}")

    inner_batch_size = int(cfg.datasets.wam_ttt.inner_batch_size)
    outer_batch_size = int(cfg.datasets.wam_ttt.outer_batch_size)
    human_sampler = DistributedTaskBatchSampler(
        dataset,
        batch_size=inner_batch_size,
        tasks=["video_gen"],
        task_weights={"video_gen": 1.0},
        seed=int(cfg.seed),
        drop_last=True,
    )
    outer_tasks = list(cfg.datasets.wam_ttt.outer_tasks)
    robot_sampler = DistributedTaskBatchSampler(
        dataset,
        batch_size=outer_batch_size,
        tasks=outer_tasks,
        task_weights={task: 1.0 for task in outer_tasks},
        seed=int(cfg.seed) + 1,
        drop_last=True,
    )
    workers = int(cfg.datasets.wam_ttt.num_workers)
    human_loader = DataLoader(dataset, batch_sampler=human_sampler, collate_fn=collate_fn, num_workers=workers)
    robot_loader = DataLoader(dataset, batch_sampler=robot_sampler, collate_fn=collate_fn, num_workers=workers)
    if not human_loader or not robot_loader:
        raise ValueError("Dataset is too small for the configured batch sizes and distributed world size")
    return human_loader, robot_loader


def _next_batch(iterator, loader, epoch):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        sampler = loader.batch_sampler
        if callable(getattr(sampler, "set_epoch", None)):
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def main(cfg) -> None:
    precision = str(cfg.trainer.mixed_precision)
    if precision not in {"bf16", "no"}:
        raise ValueError("Manual WAM-TTT gradient synchronization currently supports bf16 or no autocast, not fp16")
    data_root = Path(cfg.datasets.vla_data.data_root_dir).expanduser()
    if not data_root.is_dir():
        raise FileNotFoundError(f"LeRobot data root does not exist: {data_root}")
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.trainer.gradient_accumulation_steps),
        mixed_precision=str(cfg.trainer.mixed_precision),
    )
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output_dir / "config.yaml")
    accelerator.wait_for_everyone()
    torch.manual_seed(int(cfg.seed))

    model = build_framework(cfg)
    if bool(cfg.trainer.gradient_checkpointing):
        vlm = model.qwen_vl_interface.model
        vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(vlm.config, "use_cache"):
            vlm.config.use_cache = False
    pretrained = cfg.trainer.get("pretrained_checkpoint")
    if pretrained:
        state = torch.load(pretrained, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        accelerator.print(f"Loaded {pretrained}; missing={len(missing)}, unexpected={len(unexpected)}")

    if cfg.trainer.get("freeze_modules", ""):
        from lda.training.trainer_utils.trainer_tools import TrainerUtils

        TrainerUtils.freeze_backbones(model, cfg.trainer.freeze_modules)

    model = model.to(accelerator.device)
    human_loader, robot_loader = build_task_loaders(cfg)
    param_groups = build_param_lr_groups(model, cfg)
    optimizer_kwargs = {
        "lr": float(cfg.trainer.learning_rate.base),
        "betas": tuple(cfg.trainer.optimizer.betas),
        "eps": float(cfg.trainer.optimizer.eps),
        "weight_decay": float(cfg.trainer.optimizer.weight_decay),
    }
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        optimizer = ZeroRedundancyOptimizer(
            param_groups,
            optimizer_class=torch.optim.AdamW,
            **optimizer_kwargs,
        )
    else:
        optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
    scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(cfg.trainer.num_warmup_steps),
        num_training_steps=int(cfg.trainer.max_train_steps),
        scheduler_specific_kwargs=OmegaConf.to_container(cfg.trainer.scheduler_specific_kwargs, resolve=True),
    )
    resume_from = cfg.trainer.get("resume_from")
    start_step = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
    learner = WAMTTTMetaLearner(
        model,
        inner_steps=int(cfg.wam_ttt.inner_steps),
        inner_learning_rate=float(cfg.wam_ttt.meta_learning_rate),
        test_learning_rate=float(cfg.wam_ttt.test_learning_rate),
        kv_loss_weight=float(cfg.wam_ttt.kv_loss_weight),
    )

    step = start_step
    human_epoch = step // max(len(human_loader), 1)
    robot_epoch = step // max(len(robot_loader), 1)
    human_loader.batch_sampler.set_epoch(human_epoch)
    robot_loader.batch_sampler.set_epoch(robot_epoch)
    human_iterator, robot_iterator = iter(human_loader), iter(robot_loader)
    optimizer.zero_grad(set_to_none=True)
    while step < int(cfg.trainer.max_train_steps):
        for _ in range(int(cfg.datasets.wam_ttt.pair_retry_batches)):
            human_batch, human_iterator, human_epoch = _next_batch(human_iterator, human_loader, human_epoch)
            robot_batch, robot_iterator, robot_epoch = _next_batch(robot_iterator, robot_loader, robot_epoch)
            try:
                human_batch, robot_batch = phase_align_human_examples(human_batch, robot_batch)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(
                "No task-matched human/robot batch was found after "
                f"{cfg.datasets.wam_ttt.pair_retry_batches} batch attempts. "
                "Check the language labels or narrow the configured dataset mixture."
            )

        with accelerator.accumulate(model):
            with accelerator.autocast():
                result = learner(human_batch, robot_batch)
            accelerator.backward(result["loss"])
            if accelerator.sync_gradients:
                _average_gradients(model)
                if cfg.trainer.gradient_clipping is not None:
                    accelerator.clip_grad_norm_(model.parameters(), float(cfg.trainer.gradient_clipping))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            step += 1
            if step % int(cfg.trainer.logging_frequency) == 0:
                metrics = {
                    key: float(result[key].detach().float().item())
                    for key in ("loss", "action_loss", "dynamics_loss", "inner_video_loss", "inner_kv_loss")
                }
                accelerator.print(f"step={step} {metrics}")
            if step % int(cfg.trainer.save_interval) == 0:
                _save_checkpoint(accelerator, model, optimizer, scheduler, output_dir / "checkpoints" / f"step_{step}", step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(model.state_dict(), output_dir / "final_model.pt")
    accelerator.end_training()


def _average_gradients(model: torch.nn.Module) -> None:
    """Synchronize meta-gradients across ranks without DDP's autograd.grad hooks."""
    if not dist.is_available() or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    device = parameters[0].device
    presence = torch.tensor([parameter.grad is not None for parameter in parameters], device=device, dtype=torch.int32)
    dist.all_reduce(presence, op=dist.ReduceOp.SUM)
    groups: dict[tuple[torch.device, torch.dtype], list[torch.nn.Parameter]] = {}
    for parameter, count in zip(parameters, presence.tolist(), strict=True):
        if count == 0:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        groups.setdefault((parameter.device, parameter.grad.dtype), []).append(parameter)

    for (_, dtype), grouped_parameters in groups.items():
        max_elements = max(1, (25 * 1024 * 1024) // torch.empty((), dtype=dtype).element_size())
        bucket = []
        bucket_size = 0
        for parameter in grouped_parameters:
            if parameter.numel() > max_elements:
                if bucket:
                    _average_gradient_bucket(bucket, world_size)
                    bucket, bucket_size = [], 0
                for start in range(0, parameter.numel(), max_elements):
                    gradient_slice = parameter.grad.reshape(-1)[start : start + max_elements]
                    dist.all_reduce(gradient_slice, op=dist.ReduceOp.SUM)
                    gradient_slice.div_(world_size)
                continue
            if bucket and bucket_size + parameter.numel() > max_elements:
                _average_gradient_bucket(bucket, world_size)
                bucket, bucket_size = [], 0
            bucket.append(parameter)
            bucket_size += parameter.numel()
        if bucket:
            _average_gradient_bucket(bucket, world_size)


def _average_gradient_bucket(parameters: list[torch.nn.Parameter], world_size: int) -> None:
    flat = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.grad.copy_(flat[offset : offset + size].view_as(parameter))
        offset += size


def _save_checkpoint(accelerator, model, optimizer, scheduler, path: Path, step: int) -> None:
    accelerator.wait_for_everyone()
    if callable(getattr(optimizer, "consolidate_state_dict", None)):
        optimizer.consolidate_state_dict(to=0)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            path / "training_state.pt",
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default="lda/config/training/WAM_TTT_LDA.yaml")
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_yaml), OmegaConf.from_dotlist(normalize_dotlist_args(overrides)))
    main(cfg)
