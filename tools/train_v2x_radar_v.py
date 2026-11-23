#!/usr/bin/env python3
"""Clean single-GPU R4Det training CLI for V2X-Radar-V (Stages 3 & 4).

Unlike ``tools/train_vod.py``, this entry point:
- NEVER overwrites repository source files via ``shutil.copy``.
- Supports warm-start (``--load-from``) and full checkpoint resume
  (``--resume-from`` restoring epoch, iter, optimizer state, and LR schedule).
- Logs per-step loss, LR, finite gradient norm, peak GPU memory, and
  parameter-update L2 norm to ``work_dir/train_metrics.jsonl`` and
  ``work_dir/train_summary.json`` for Stage 3/4/5 verification and plotting.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/r4det-v2x-train-mpl")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer, load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train R4Det on V2X-Radar-V (Stage 3 small-sample & Stage 4 full)."
    )
    parser.add_argument(
        "--config",
        default="configs/r4det/V2X-Radar-V-R4Det_single_frame.py",
        help="Path to V2X-Radar-V config.",
    )
    parser.add_argument(
        "--work-dir",
        default="work_dirs/v2x_radar_v/train_run",
        help="Output directory for checkpoints, logs, and metrics.",
    )
    parser.add_argument("--ann-file-train", default=None, help="Override train ann_file.")
    parser.add_argument("--ann-file-val", default=None, help="Override val ann_file.")
    parser.add_argument("--load-from", default=None, help="Warm-start checkpoint weights.")
    parser.add_argument("--resume-from", default=None, help="Resume full training state.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max_epochs.")
    parser.add_argument("--samples-per-gpu", type=int, default=None, help="Override batch size.")
    parser.add_argument("--workers-per-gpu", type=int, default=None, help="Override num_workers.")
    parser.add_argument("--lr", type=float, default=None, help="Override base learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--deterministic", action="store_true", help="Set cudnn deterministic.")
    parser.add_argument("--no-aug", action="store_true", help="Disable random train aug (useful for Stage 3 fixed-sample overfit).")
    parser.add_argument("--filter-empty-gt", action="store_true", help="Filter out 0-box frames during training.")
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp mixed precision (bf16 on sm_80+, else fp16) to fit 8GB VRAM.")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16", help="AMP autocast dtype (bf16 avoids FP16 overflow on RTX 4060 Ti).")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps (enables effective batch >= 2 on 8GB GPUs).")
    parser.add_argument("--freeze-modules", nargs="*", default=[], help="Top-level model submodule names to freeze with no_grad (e.g. depth_net backward_projection) for 8GB GPUs.")
    parser.add_argument("--save-interval", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument(
        "--save-step-interval",
        type=int,
        default=0,
        help="Save mid-epoch latest.pth every N optimizer update steps (0 disables).",
    )
    parser.add_argument(
        "--save-time-interval",
        type=float,
        default=0.0,
        help="Save mid-epoch latest.pth whenever elapsed seconds since last save >= N (0 disables).",
    )
    parser.add_argument("--log-interval", type=int, default=1, help="Print/log every N steps.")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_param_vector_norm(model: torch.nn.Module) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            sq += float(p.detach().float().pow(2).sum().item())
    return math.sqrt(sq)


def snapshot_trainable_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().float().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def compute_param_delta_norm(
    model: torch.nn.Module, before: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
    total_sq = 0.0
    changed_tensors = 0
    total_tensors = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or name not in before:
            continue
        total_tensors += 1
        diff = p.detach().float().cpu() - before[name]
        norm_sq = float(diff.pow(2).sum().item())
        if norm_sq > 0.0:
            changed_tensors += 1
        total_sq += norm_sq
    return {
        "l2_delta_norm": round(math.sqrt(total_sq), 6),
        "changed_param_tensors": changed_tensors,
        "total_trainable_tensors": total_tensors,
    }


def parse_losses(losses: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
    log_vars: Dict[str, float] = {}
    total_loss: Optional[torch.Tensor] = None
    for name, val in losses.items():
        if torch.is_tensor(val):
            loss_t = val.mean()
        elif isinstance(val, (list, tuple)):
            loss_t = sum(v.mean() for v in val)
        else:
            continue
        log_vars[name] = round(float(loss_t.detach().cpu().item()), 6)
        if "loss" in name:
            total_loss = loss_t if total_loss is None else (total_loss + loss_t)
    if total_loss is None:
        raise RuntimeError(f"No loss tensor found in forward_train output: {list(losses.keys())}")
    log_vars["total_loss"] = round(float(total_loss.detach().cpu().item()), 6)
    return total_loss, log_vars


def save_checkpoint_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    global_step: int,
    meta: Dict[str, Any],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    step_in_epoch: int = 0,
    mid_epoch: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "meta": {
            **meta,
            "epoch": epoch,
            "iter": global_step,
            "step_in_epoch": int(step_in_epoch),
            "mid_epoch": bool(mid_epoch),
        },
    }
    if scaler is not None and scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, str(tmp_path))
    os.replace(str(tmp_path), str(path))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R4Det training.")

    logging.getLogger("mmcv").setLevel(logging.WARNING)
    set_seed(args.seed, deterministic=args.deterministic)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = work_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg.work_dir = str(work_dir)
    cfg.model.pretrained = None
    cfg.model.meta_info = {
        "figures_path": str(fig_dir),
        "project_name": f"V2X-Radar-V-{work_dir.name}",
    }

    if args.ann_file_train:
        cfg.data.train.ann_file = args.ann_file_train
    if args.ann_file_val:
        cfg.data.val.ann_file = args.ann_file_val
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
        cfg.runner.max_epochs = args.max_epochs
    if args.samples_per_gpu is not None:
        cfg.data.samples_per_gpu = args.samples_per_gpu
    if args.workers_per_gpu is not None:
        cfg.data.workers_per_gpu = args.workers_per_gpu
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    if args.filter_empty_gt:
        cfg.data.train.filter_empty_gt = True
    if args.no_aug:
        for step in cfg.data.train.pipeline:
            if step.get("type") in ("ImageAug3D2", "GlobalRotScaleTransFlipAll"):
                step["is_train"] = False

    cfg.dump(str(work_dir / Path(args.config).name))

    dataset = build_dataset(cfg.data.train)
    samples_per_gpu = int(cfg.data.get("samples_per_gpu", 1))
    workers_per_gpu = int(cfg.data.get("workers_per_gpu", 0))
    loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=not args.no_aug,
        seed=args.seed,
        drop_last=len(dataset) >= samples_per_gpu,
    )

    model = build_model(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    model.init_weights()
    model.CLASSES = dataset.CLASSES

    load_info: Dict[str, Any] = {"mode": "scratch"}
    if args.load_from and not args.resume_from:
        ckpt = torch.load(args.load_from, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        incomp = model.load_state_dict(state_dict, strict=False)
        load_info = {
            "mode": "load_from",
            "path": args.load_from,
            "missing_keys": list(incomp.missing_keys),
            "unexpected_keys": list(incomp.unexpected_keys),
        }

    model_dp = MMDataParallel(model.cuda(), device_ids=[0])
    # Freeze requested top-level modules (with no_grad forward) for 8GB GPU memory budgets
    for mod_name in args.freeze_modules:
        submod = getattr(model_dp.module, mod_name, None)
        if submod is None:
            continue
        for p in submod.parameters():
            p.requires_grad = False
        submod.eval()
        orig_mod_fwd = submod.forward
        def _make_nograd_fwd(fn):
            def _nograd_fwd(*a, **kw):
                with torch.no_grad():
                    return fn(*a, **kw)
            return _nograd_fwd
        submod.forward = _make_nograd_fwd(orig_mod_fwd)

    if args.amp and hasattr(model_dp.module, "depth_net") and model_dp.module.depth_net is not None:
        # Keep GeometryDepth_Net in FP32 to prevent FP16 overflow in depth softmax/conv;
        # when freeze_depths=True, also skip autograd graph to save ~1.8 GB VRAM on 8GB GPUs.
        orig_dn_fwd = model_dp.module.depth_net.forward
        freeze_dn = bool(getattr(model_dp.module, "freeze_depths", False))
        def _fp32_dn_fwd(inputs, radar_depth, depth_comple, img_metas):
            ctx = torch.no_grad() if freeze_dn else torch.enable_grad()
            with ctx, torch.cuda.amp.autocast(enabled=False):
                inputs_f32 = [x.float() if torch.is_tensor(x) else x for x in inputs]
                rd_f32 = radar_depth.float() if torch.is_tensor(radar_depth) else radar_depth
                dc_f32 = depth_comple.float() if torch.is_tensor(depth_comple) else depth_comple
                return orig_dn_fwd(inputs_f32, rd_f32, dc_f32, img_metas)
        model_dp.module.depth_net.forward = _fp32_dn_fwd

    if args.amp and hasattr(model_dp.module, "img_roi_head") and model_dp.module.img_roi_head is not None:
        # MMCV 1.7 RoIAlign requires float32 inputs even inside torch.cuda.amp.autocast
        for ext_name in ("bbox_roi_extractor", "mask_roi_extractor"):
            ext = getattr(model_dp.module.img_roi_head, ext_name, None)
            if ext is not None:
                orig_fwd = ext.forward
                def _make_fp32_fwd(fn):
                    def _fp32_fwd(feats, rois, *a, **kw):
                        with torch.cuda.amp.autocast(enabled=False):
                            feats_f32 = tuple(f.float() for f in feats)
                            return fn(feats_f32, rois.float(), *a, **kw)
                    return _fp32_fwd
                ext.forward = _make_fp32_fwd(orig_fwd)
    # Only pass trainable parameters to AdamW so frozen backbone/neck params do not
    # allocate unused FP32 momentum/variance states on 8GB GPUs.
    trainable_groups = []
    custom_keys = cfg.optimizer.get("paramwise_cfg", {}).get("custom_keys", {})
    base_lr = float(cfg.optimizer.lr)
    base_wd = float(cfg.optimizer.get("weight_decay", 0.01))
    default_params = []
    custom_buckets: Dict[float, List[torch.nn.Parameter]] = {}
    for pname, param in model_dp.module.named_parameters():
        if not param.requires_grad:
            continue
        matched_mult = None
        for key_prefix, key_cfg in custom_keys.items():
            if key_prefix in pname:
                matched_mult = float(key_cfg.get("lr_mult", 1.0))
                break
        if matched_mult is not None:
            custom_buckets.setdefault(matched_mult, []).append(param)
        else:
            default_params.append(param)
    trainable_groups.append({"params": default_params, "lr": base_lr, "weight_decay": base_wd})
    for mult, plist in custom_buckets.items():
        trainable_groups.append({"params": plist, "lr": base_lr * mult, "weight_decay": base_wd})
    optimizer = torch.optim.AdamW(
        trainable_groups,
        lr=base_lr,
        betas=tuple(cfg.optimizer.get("betas", (0.95, 0.99))),
        weight_decay=base_wd,
    )
    max_epochs = int(cfg.runner.max_epochs)
    min_lr = float(cfg.optimizer.lr) * float(cfg.lr_config.get("min_lr_ratio", 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_epochs, 1), eta_min=min_lr
    )

    amp_dtype = torch.bfloat16 if (args.amp_dtype == "bf16" and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = bool(args.amp and amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    start_epoch = 0
    global_step = 0
    resume_step_in_epoch = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        incomp = model_dp.module.load_state_dict(ckpt["state_dict"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if use_scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        meta = ckpt.get("meta", {})
        start_epoch = int(meta.get("epoch", 0))
        global_step = int(meta.get("iter", 0))
        if bool(meta.get("mid_epoch", False)):
            resume_step_in_epoch = int(meta.get("step_in_epoch", 0))
        load_info = {
            "mode": "resume_from",
            "path": args.resume_from,
            "resumed_epoch": start_epoch,
            "resumed_step_in_epoch": resume_step_in_epoch,
            "resumed_mid_epoch": bool(meta.get("mid_epoch", False)),
            "resumed_iter": global_step,
            "resumed_lr": float(optimizer.param_groups[0]["lr"]),
            "missing_keys": list(incomp.missing_keys),
            "unexpected_keys": list(incomp.unexpected_keys),
        }
        print(
            f"[Resume] path={args.resume_from} start_epoch={start_epoch + 1}/{max_epochs} "
            f"step_in_epoch={resume_step_in_epoch} iter={global_step} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}",
            flush=True,
        )

    # Snapshot trainable weights before training to verify real parameter updates
    params_before = snapshot_trainable_params(model_dp.module)
    metrics_path = work_dir / "train_metrics.jsonl"
    grad_clip_norm = float(cfg.optimizer_config.get("grad_clip", {}).get("max_norm", 35.0))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    epoch_losses: List[Dict[str, Any]] = []
    all_grads_finite = True
    any_nan_loss = False
    t_start = time.perf_counter()
    last_save_time = t_start
    ckpt_meta = {
        "CLASSES": list(dataset.CLASSES),
        "config": args.config,
        "seed": args.seed,
    }

    model_dp.train()
    if samples_per_gpu == 1:
        # BatchNorm layers on (B, C) / (B, C, 1, 1) tensors (camera MLP & ASPP global_avg_pool)
        # require B > 1 in train mode; keep BN running stats in eval mode for micro-batch=1 + grad_accum.
        for module in model_dp.module.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    grad_accum = max(1, int(args.grad_accum_steps))
    total_batches = len(loader)
    for epoch in range(start_epoch, max_epochs):
        ep_losses: List[float] = []
        ep_grad_norms: List[float] = []
        ep_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        skip_steps = resume_step_in_epoch if (epoch == start_epoch and resume_step_in_epoch > 0) else 0
        if skip_steps > 0:
            all_batches = list(loader.batch_sampler)
            remaining_batches = all_batches[skip_steps:]
            print(
                f"[Resume] Fast-forwarding Epoch {epoch + 1}/{max_epochs}: "
                f"skipping {skip_steps}/{len(all_batches)} batches "
                f"({len(remaining_batches)} remaining)",
                flush=True,
            )
            epoch_loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=remaining_batches,
                num_workers=loader.num_workers,
                collate_fn=loader.collate_fn,
                pin_memory=loader.pin_memory,
                worker_init_fn=loader.worker_init_fn,
            )
            resume_step_in_epoch = 0
        else:
            epoch_loader = loader

        for offset, batch in enumerate(epoch_loader):
            step_idx = skip_steps + offset
            with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype):
                losses = model_dp(return_loss=True, **batch)
                total_loss, log_vars = parse_losses(losses)

            if not torch.isfinite(total_loss):
                any_nan_loss = True
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1} step {step_idx + 1}: {log_vars}")

            scaled_loss = total_loss / float(grad_accum)
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_update_step = ((step_idx + 1) % grad_accum == 0) or ((step_idx + 1) == total_batches)
            grad_norm = 0.0
            if is_update_step:
                if use_scaler:
                    scaler.unscale_(optimizer)
                grad_norm_t = torch.nn.utils.clip_grad_norm_(
                    [p for p in model_dp.module.parameters() if p.requires_grad],
                    max_norm=grad_clip_norm,
                )
                grad_norm = float(grad_norm_t.detach().cpu().item())
                if not math.isfinite(grad_norm):
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    all_grads_finite = False
                    raise RuntimeError(f"Non-finite grad_norm={grad_norm} at epoch {epoch + 1} step {step_idx + 1}")

                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            lr_now = float(optimizer.param_groups[0]["lr"])
            ep_losses.append(log_vars["total_loss"])
            if is_update_step:
                ep_grad_norms.append(grad_norm)

            if is_update_step:
                record = {
                    "epoch": epoch + 1,
                    "step_in_epoch": step_idx + 1,
                    "iter": global_step,
                    "lr": lr_now,
                    "grad_norm": round(grad_norm, 4),
                    "peak_mem_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
                    **log_vars,
                }
                with open(metrics_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(record) + "\n")

                if global_step % args.log_interval == 0:
                    print(
                        f"[Epoch {epoch + 1}/{max_epochs} | Step {step_idx + 1}/{total_batches} | Iter {global_step}] "
                        f"loss={log_vars['total_loss']:.4f} lr={lr_now:.6g} "
                        f"grad_norm={grad_norm:.3f} mem={record['peak_mem_mib']}MiB",
                        flush=True,
                    )

                now = time.perf_counter()
                due_by_step = args.save_step_interval > 0 and (global_step % args.save_step_interval == 0)
                due_by_time = args.save_time_interval > 0.0 and ((now - last_save_time) >= args.save_time_interval)
                if (due_by_step or due_by_time) and ((step_idx + 1) < total_batches):
                    latest_path = work_dir / "latest.pth"
                    save_checkpoint_state(
                        latest_path,
                        model_dp.module,
                        optimizer,
                        scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        meta=ckpt_meta,
                        scaler=scaler,
                        step_in_epoch=step_idx + 1,
                        mid_epoch=True,
                    )
                    last_save_time = time.perf_counter()
                    print(
                        f"[Checkpoint] Saved mid-epoch {latest_path} "
                        f"(epoch={epoch + 1}/{max_epochs}, step={step_idx + 1}/{total_batches}, iter={global_step})",
                        flush=True,
                    )

        scheduler.step()
        ep_mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        ep_mean_grad = float(np.mean(ep_grad_norms)) if ep_grad_norms else float("nan")
        epoch_losses.append(
            {
                "epoch": epoch + 1,
                "mean_loss": round(ep_mean_loss, 6),
                "mean_grad_norm": round(ep_mean_grad, 4),
                "lr_after_step": float(optimizer.param_groups[0]["lr"]),
                "elapsed_sec": round(time.perf_counter() - ep_t0, 3),
            }
        )

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == max_epochs:
            ckpt_path = work_dir / f"epoch_{epoch + 1}.pth"
            save_checkpoint_state(
                ckpt_path,
                model_dp.module,
                optimizer,
                scheduler,
                epoch=epoch + 1,
                global_step=global_step,
                meta=ckpt_meta,
                scaler=scaler,
                step_in_epoch=0,
                mid_epoch=False,
            )
            latest_path = work_dir / "latest.pth"
            save_checkpoint_state(
                latest_path,
                model_dp.module,
                optimizer,
                scheduler,
                epoch=epoch + 1,
                global_step=global_step,
                meta=ckpt_meta,
                scaler=scaler,
                step_in_epoch=0,
                mid_epoch=False,
            )
            last_save_time = time.perf_counter()
            print(
                f"[Checkpoint] Saved {ckpt_path} & {latest_path} "
                f"(completed epoch={epoch + 1}/{max_epochs}, iter={global_step})",
                flush=True,
            )

    param_delta = compute_param_delta_norm(model_dp.module, params_before)
    summary = {
        "status": "ok",
        "config": args.config,
        "work_dir": str(work_dir),
        "train_ann_file": cfg.data.train.ann_file,
        "dataset_len": len(dataset),
        "samples_per_gpu": samples_per_gpu,
        "start_epoch": start_epoch,
        "end_epoch": max_epochs,
        "final_iter": global_step,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "device": torch.cuda.get_device_name(0),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "elapsed_seconds": round(time.perf_counter() - t_start, 3),
        "load_info": load_info,
        "all_grads_finite": all_grads_finite,
        "any_nan_loss": any_nan_loss,
        "param_update": param_delta,
        "epoch_losses": epoch_losses,
        "latest_checkpoint": str(work_dir / "latest.pth"),
        "metrics_jsonl": str(metrics_path),
    }
    summary_path = work_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"TRAIN_OK summary={summary_path} param_delta_l2={param_delta['l2_delta_norm']} "
        f"peak_mem={summary['peak_gpu_memory_mib']}MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
