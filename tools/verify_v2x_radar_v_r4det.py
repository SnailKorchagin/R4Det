#!/usr/bin/env python3
"""Stage 2 end-to-end verification for V2X-Radar-V in R4Det.

Verifies:
1. Dataset & DataLoader construction via ``build_dataset`` / ``build_dataloader``
   from ``configs/r4det/V2X-Radar-V-R4Det_single_frame.py``.
2. Real labeled frame (with boxes), real legitimate empty-GT frame (000359),
   unlabeled inference path (annos=None), and missing-input error handling.
3. Post-augmentation geometry consistency (radar_depth & BEV axes).
4. Real single-frame GPU forward with TJ4D weights (cross-dataset check),
   stage-by-stage hook trace, GT removal/perturbation audit in ``simple_test``,
   prediction JSON, and BEV visualization.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/r4det-v2x-mpl")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import DataContainer, MMDataParallel
from mmcv.runner import load_checkpoint

from code_walkthrough.generate_r4det_storyboard import TraceRecorder
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.datasets.v2x_radar_v_utils import (
    format_radar_points_for_r4det,
    normalize_frame_id,
    parse_calib_file,
    read_radar_bin,
    resolve_frame_files,
)
from mmdet3d.models import build_model
from tools.r4det_smoke_test import describe_batch_value, unwrap_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify V2X-Radar-V dataset adapter, batch loading, and GPU forward in R4Det."
    )
    parser.add_argument(
        "--config",
        default="configs/r4det/V2X-Radar-V-R4Det_single_frame.py",
        help="Path to V2X-Radar-V R4Det config.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/final_tj4d.pth",
        help="Checkpoint for Stage 2 cross-dataset GPU forward check.",
    )
    parser.add_argument(
        "--frame-id",
        default="000009",
        help="Target validation frame ID for single-frame forward & BEV plot.",
    )
    parser.add_argument(
        "--empty-frame-id",
        default="000359",
        help="Known legitimate empty-GT frame ID in train split.",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/v2x_radar_v/stage2",
        help="Directory for prediction.json, stage_trace.json, and bev_pred.png.",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip GPU forward even if CUDA is available.",
    )
    return parser.parse_args()


def _get_sample0_pair(batch: Dict[str, Any], key: str):
    """Return [prev, curr] for sample 0 from MMCV DataLoader collated DataContainer."""
    val = batch[key].data[0]
    if torch.is_tensor(val):
        return val[0]
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list) and len(val[0]) == 2:
        return val[0]
    return val


def summarize_batch(batch: Dict[str, Any], class_names: List[str]) -> Dict[str, Any]:
    """Extract compact, JSON-serializable summary of one R4Det batch."""
    pts_pair = _get_sample0_pair(batch, "points")  # [prev_pts, curr_pts]
    img_pair = _get_sample0_pair(batch, "img")  # [2, C, H, W]
    metas_pair = _get_sample0_pair(batch, "img_metas")  # [prev_meta, curr_meta]
    gt_boxes_pair = _get_sample0_pair(batch, "gt_bboxes_3d")
    gt_labels_pair = _get_sample0_pair(batch, "gt_labels_3d")
    gt_2d_pair = _get_sample0_pair(batch, "gt_bboxes")

    curr_pts = pts_pair[1]
    curr_meta = metas_pair[1]
    prev_meta = metas_pair[0]
    curr_boxes = gt_boxes_pair[1].tensor
    curr_labels = gt_labels_pair[1]

    pts_np = curr_pts.detach().cpu().numpy() if torch.is_tensor(curr_pts) else np.asarray(curr_pts)
    pts_stats = None
    if pts_np.shape[0] > 0:
        pts_stats = {
            "x_velo": [round(float(pts_np[:, 0].min()), 3), round(float(pts_np[:, 0].max()), 3)],
            "y_velo": [round(float(pts_np[:, 1].min()), 3), round(float(pts_np[:, 1].max()), 3)],
            "z_velo": [round(float(pts_np[:, 2].min()), 3), round(float(pts_np[:, 2].max()), 3)],
            "velocity": [round(float(pts_np[:, 3].min()), 3), round(float(pts_np[:, 3].max()), 3)],
            "intensity": [round(float(pts_np[:, 4].min()), 3), round(float(pts_np[:, 4].max()), 3)],
        }

    radar_depth = curr_meta.get("radar_depth")
    radar_depth_nonzero = int(np.count_nonzero(radar_depth > 0)) if radar_depth is not None else 0
    label_list = curr_labels.detach().cpu().tolist() if torch.is_tensor(curr_labels) else list(curr_labels)
    mapped_class_counts: Dict[str, int] = {}
    for lbl in label_list:
        cname = class_names[lbl] if 0 <= lbl < len(class_names) else "ignore(-1)"
        mapped_class_counts[cname] = mapped_class_counts.get(cname, 0) + 1

    return {
        "frame_id": curr_meta.get("frame_id", str(curr_meta.get("sample_idx"))),
        "batch_keys": sorted(batch.keys()),
        "shapes": {k: describe_batch_value(v.data) for k, v in batch.items()},
        "prev_frame_valid": bool(prev_meta.get("is_prev_frame_valid", False)),
        "curr_frame_valid": bool(curr_meta.get("is_prev_frame_valid", False)),
        "ori_shape": list(curr_meta.get("ori_shape", [])),
        "img_shape": list(curr_meta.get("img_shape", [])),
        "pad_shape": list(curr_meta.get("pad_shape", [])),
        "curr_points_count": int(pts_np.shape[0]),
        "curr_points_dim": int(pts_np.shape[1]) if pts_np.ndim == 2 else 0,
        "curr_points_ranges": pts_stats,
        "radar_intensity_cfg": curr_meta.get("radar_intensity_cfg"),
        "radar_depth_shape": list(radar_depth.shape) if radar_depth is not None else None,
        "radar_depth_nonzero_pixels": radar_depth_nonzero,
        "segmentation_source": curr_meta.get("segmentation_source"),
        "gt_depths_source": curr_meta.get("gt_depths_source"),
        "has_annotations": curr_meta.get("has_annotations"),
        "gt_bboxes_3d_shape": list(curr_boxes.shape),
        "gt_bboxes_2d_shape": list(gt_2d_pair[1].shape),
        "gt_count": int(curr_boxes.shape[0]),
        "gt_class_counts": mapped_class_counts,
        "img_tensor_shape": list(img_pair.shape),
    }


def check_unlabeled_and_missing_inputs(cfg: Config, val_dataset) -> Dict[str, Any]:
    """Verify (1) unlabeled path (annos=None) and (2) missing critical files raise FileNotFoundError."""
    results: Dict[str, Any] = {}
    sample_info = copy.deepcopy(val_dataset.data_infos[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Unlabeled path (annos=None, allow_unlabeled=True)
        unlabeled_info = copy.deepcopy(sample_info)
        unlabeled_info["annos"] = None
        unlabeled_pkl = tmp / "unlabeled.pkl"
        with open(unlabeled_pkl, "wb") as f:
            pickle.dump([unlabeled_info], f)

        unlabeled_cfg = copy.deepcopy(cfg.data.val)
        unlabeled_cfg.ann_file = str(unlabeled_pkl)
        unlabeled_cfg.allow_unlabeled = True
        unlabeled_ds = build_dataset(unlabeled_cfg)
        unlabeled_loader = build_dataloader(
            unlabeled_ds, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
        )
        unlabeled_batch = next(iter(unlabeled_loader))
        unlabeled_summary = summarize_batch(unlabeled_batch, list(unlabeled_ds.CLASSES))
        results["unlabeled_path"] = {
            "passed": (
                unlabeled_summary["curr_points_count"] > 0
                and unlabeled_summary["gt_count"] == 0
                and unlabeled_summary["has_annotations"] is False
            ),
            "summary": unlabeled_summary,
        }

        # 2. Missing radar file raises FileNotFoundError
        missing_radar_info = copy.deepcopy(sample_info)
        missing_radar_info["point_cloud"]["radar_path"] = "training/radar/999999.bin"
        missing_radar_info["point_cloud"]["velodyne_path"] = "training/radar/999999.bin"
        missing_radar_pkl = tmp / "missing_radar.pkl"
        with open(missing_radar_pkl, "wb") as f:
            pickle.dump([missing_radar_info], f)

        mr_cfg = copy.deepcopy(cfg.data.val)
        mr_cfg.ann_file = str(missing_radar_pkl)
        mr_ds = build_dataset(mr_cfg)
        try:
            _ = mr_ds[0]
            results["missing_radar_raises"] = {"passed": False, "error": None}
        except FileNotFoundError as exc:
            results["missing_radar_raises"] = {"passed": True, "error": str(exc)}

        # 3. Missing image file raises FileNotFoundError
        missing_img_info = copy.deepcopy(sample_info)
        missing_img_info["image"]["image_path"] = "training/image_2/999999.jpg"
        missing_img_pkl = tmp / "missing_img.pkl"
        with open(missing_img_pkl, "wb") as f:
            pickle.dump([missing_img_info], f)

        mi_cfg = copy.deepcopy(cfg.data.val)
        mi_cfg.ann_file = str(missing_img_pkl)
        mi_ds = build_dataset(mi_cfg)
        try:
            _ = mi_ds[0]
            results["missing_image_raises"] = {"passed": False, "error": None}
        except FileNotFoundError as exc:
            results["missing_image_raises"] = {"passed": True, "error": str(exc)}

    return results


def _clone_batch_with_gt_mode(batch: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Create a deep copy of a batch with 'original', 'empty', or 'perturbed' GT."""
    out = copy.deepcopy(batch)
    if mode == "original":
        return out

    gt_boxes_dc = out["gt_bboxes_3d"]
    gt_labels_dc = out["gt_labels_3d"]
    gt_2d_dc = out["gt_bboxes"]
    gt_2d_lbl_dc = out["gt_labels"]

    prev_b3, curr_b3 = _get_sample0_pair(out, "gt_bboxes_3d")
    prev_l3, curr_l3 = _get_sample0_pair(out, "gt_labels_3d")
    prev_b2, curr_b2 = _get_sample0_pair(out, "gt_bboxes")
    prev_l2, curr_l2 = _get_sample0_pair(out, "gt_labels")

    if mode == "empty":
        empty_b3 = LiDARInstance3DBoxes(torch.zeros((0, 7), dtype=curr_b3.tensor.dtype))
        empty_l3 = torch.zeros((0,), dtype=curr_l3.dtype)
        empty_b2 = torch.zeros((0, 4), dtype=curr_b2.dtype)
        empty_l2 = torch.zeros((0,), dtype=curr_l2.dtype)
        out["gt_bboxes_3d"].data[0][0] = [prev_b3, empty_b3]
        out["gt_labels_3d"].data[0][0] = [prev_l3, empty_l3]
        out["gt_bboxes"].data[0][0] = [prev_b2, empty_b2]
        out["gt_labels"].data[0][0] = [prev_l2, empty_l2]
    elif mode == "perturbed":
        pert_tensor = curr_b3.tensor.clone()
        if pert_tensor.shape[0] > 0:
            pert_tensor[:, 0] += 20.0
            pert_tensor[:, 1] += 10.0
            pert_tensor[:, 6] += 1.57
        pert_b3 = LiDARInstance3DBoxes(pert_tensor)
        pert_l3 = (curr_l3 + 1) % 4
        pert_b2 = curr_b2.clone()
        if pert_b2.shape[0] > 0:
            pert_b2[:, [0, 2]] = torch.clamp(pert_b2[:, [0, 2]] + 80.0, 0.0, 640.0)
        pert_l2 = (curr_l2 + 1) % 4
        out["gt_bboxes_3d"].data[0][0] = [prev_b3, pert_b3]
        out["gt_labels_3d"].data[0][0] = [prev_l3, pert_l3]
        out["gt_bboxes"].data[0][0] = [prev_b2, pert_b2]
        out["gt_labels"].data[0][0] = [prev_l2, pert_l2]
    else:
        raise ValueError(f"Unknown GT mode: {mode}")

    return out


def _compare_predictions(ref_out, alt_out) -> Dict[str, Any]:
    ref_p = ref_out[0]["pts_bbox"] if "pts_bbox" in ref_out[0] else ref_out[0]
    alt_p = alt_out[0]["pts_bbox"] if "pts_bbox" in alt_out[0] else alt_out[0]

    rb = ref_p["boxes_3d"].tensor.detach().cpu()
    ab = alt_p["boxes_3d"].tensor.detach().cpu()
    rs = ref_p["scores_3d"].detach().cpu()
    as_ = alt_p["scores_3d"].detach().cpu()
    rl = ref_p["labels_3d"].detach().cpu()
    al = alt_p["labels_3d"].detach().cpu()

    same_count = rb.shape == ab.shape
    if not same_count:
        return {"exact_match": False, "ref_count": int(rs.numel()), "alt_count": int(as_.numel())}
    if rs.numel() == 0:
        return {"exact_match": True, "ref_count": 0, "alt_count": 0, "max_box_diff": 0.0, "max_score_diff": 0.0}

    # Compare sorted top-10 (avoids low-score tail tie-break flips from CUDA atomicAdd)
    ref_ord = torch.argsort(rs, descending=True)[:10]
    alt_ord = torch.argsort(as_, descending=True)[:10]
    top10_box_diff = float((rb[ref_ord] - ab[alt_ord]).abs().max().item())
    top10_score_diff = float((rs[ref_ord] - as_[alt_ord]).abs().max().item())
    top10_labels_equal = bool(torch.equal(rl[ref_ord], al[alt_ord]))

    max_box_diff = float((rb - ab).abs().max().item())
    max_score_diff = float((rs - as_).abs().max().item())
    labels_equal = bool(torch.equal(rl, al))
    return {
        "exact_match": bool(max_box_diff == 0.0 and max_score_diff == 0.0 and labels_equal),
        "top10_close": bool(top10_box_diff < 1e-3 and top10_score_diff < 1e-4 and top10_labels_equal),
        "ref_count": int(rs.numel()),
        "alt_count": int(as_.numel()),
        "top10_max_box_diff": round(top10_box_diff, 7),
        "top10_max_score_diff": round(top10_score_diff, 7),
        "top10_labels_equal": top10_labels_equal,
        "all300_max_box_diff": round(max_box_diff, 7),
        "all300_max_score_diff": round(max_score_diff, 7),
        "all300_labels_equal": labels_equal,
    }


def render_v2x_bev(
    data_root: Path,
    frame_id: str,
    dataset,
    sample_index: int,
    predictions: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    """Render Velodyne-frame V2X-Radar-V points, GT 3D boxes, and predictions in BEV."""
    height, width = 900, 900
    x_min, x_max = 0.0, 69.12
    y_min, y_max = -39.68, 39.68
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)

    def pixel(xy):
        x, y = xy
        col = int((y_max - y) / (y_max - y_min) * (width - 1))
        row = int((x_max - x) / (x_max - x_min) * (height - 1))
        return col, row

    files = resolve_frame_files(data_root, frame_id)
    calib = parse_calib_file(files["calib"])
    radar_raw = read_radar_bin(files["radar"], num_features=5)
    radar_velo = format_radar_points_for_r4det(radar_raw, calib["Tr_radar_to_velo"])

    valid = (
        (radar_velo[:, 0] >= x_min)
        & (radar_velo[:, 0] <= x_max)
        & (radar_velo[:, 1] >= y_min)
        & (radar_velo[:, 1] <= y_max)
    )
    for x, y in radar_velo[valid, :2]:
        cv2.circle(canvas, pixel((x, y)), 1, (160, 160, 160), -1)

    def draw_box(box, color, text):
        x, y, _, dx, dy, _, yaw = box[:7]
        local = np.array(
            [[dx / 2, dy / 2], [dx / 2, -dy / 2], [-dx / 2, -dy / 2], [-dx / 2, dy / 2]],
            dtype=np.float32,
        )
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float32)
        corners = local @ rot.T + np.array([x, y], dtype=np.float32)
        poly = np.array([pixel(pt) for pt in corners], dtype=np.int32)
        cv2.polylines(canvas, [poly], True, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, text, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    gt = dataset.get_ann_info(sample_index)
    gt_boxes = gt["gt_bboxes_3d"].tensor.cpu().numpy()
    for box, label in zip(gt_boxes, gt["gt_labels_3d"]):
        name = dataset.CLASSES[int(label)] if 0 <= int(label) < len(dataset.CLASSES) else "ignore"
        draw_box(box, (40, 75, 255), f"GT {name}")

    for pred in predictions:
        lbl = int(pred["label"])
        name = dataset.CLASSES[lbl] if 0 <= lbl < len(dataset.CLASSES) else str(lbl)
        draw_box(pred["box"], (70, 235, 70), f"P {name} {pred['score']:.3f}")

    cv2.putText(
        canvas,
        f"R4Det x V2X-Radar-V frame {frame_id} BEV (TJ4D cross-dataset weights)",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "gray=4D radar (velo frame)  red=GT  green=top10 pred",
        (18, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (215, 215, 215),
        1,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f"Failed to write BEV visualization to {out_path}")


def main() -> None:
    args = parse_args()
    logging.getLogger("mmcv").setLevel(logging.WARNING)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cfg.model.meta_info = {
        "figures_path": str(fig_dir),
        "project_name": "V2X-Radar-V-R4Det-stage2",
    }

    # 1. Build train & val datasets via standard R4Det build_dataset / build_dataloader
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)

    train_loader = build_dataloader(
        train_dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
    )
    val_loader = build_dataloader(
        val_dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
    )

    # Locate target frames in train & val
    target_fid = normalize_frame_id(args.frame_id)
    empty_fid = normalize_frame_id(args.empty_frame_id)

    train_batches: Dict[str, Dict[str, Any]] = {}
    for batch in train_loader:
        fid = _get_sample0_pair(batch, "img_metas")[1].get("frame_id")
        if fid not in train_batches and (len(train_batches) == 0 or fid == empty_fid):
            train_batches[fid] = summarize_batch(batch, list(train_dataset.CLASSES))

    val_target_batch = None
    val_target_index = 0
    for idx, batch in enumerate(val_loader):
        fid = _get_sample0_pair(batch, "img_metas")[1].get("frame_id")
        if fid == target_fid:
            val_target_batch = batch
            val_target_index = idx
            break
    if val_target_batch is None:
        raise ValueError(
            f"Target frame {target_fid} not found in val dataset ({cfg.data.val.ann_file})."
        )
    val_summary = summarize_batch(val_target_batch, list(val_dataset.CLASSES))

    edge_checks = check_unlabeled_and_missing_inputs(cfg, val_dataset)

    print(f"BATCH_VERIFY_OK val_frame={target_fid} pts={val_summary['curr_points_count']} gt={val_summary['gt_count']} prev_valid={val_summary['prev_frame_valid']}")
    if empty_fid in train_batches:
        eb = train_batches[empty_fid]
        print(f"EMPTY_GT_OK train_frame={empty_fid} pts={eb['curr_points_count']} gt={eb['gt_count']}")
    print(f"EDGE_CHECKS unlabeled={edge_checks['unlabeled_path']['passed']} missing_radar={edge_checks['missing_radar_raises']['passed']} missing_img={edge_checks['missing_image_raises']['passed']}")

    prediction_report: Dict[str, Any] = {
        "status": "batch_only",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "target_frame_id": target_fid,
        "input_sources": {
            "camera_image": "original_v2x_radar_v_jpg",
            "radar_points": "v2x_radar_v_5d_transformed_to_velo_reordered_xyz_vel_intensity",
            "intensity_definition": "raw col 3 -> R4Det col 4 with offset=0.0, scale=1.0",
            "segmentation_source": val_summary["segmentation_source"],
            "temporal_mode": "single_frame (use_temporal=False, is_prev_frame_valid=False)",
        },
        "dataset_verification": {
            "train_dataset_len": len(train_dataset),
            "val_dataset_len": len(val_dataset),
            "train_sample_summaries": train_batches,
            "val_target_summary": val_summary,
            "edge_checks": edge_checks,
        },
    }

    if args.skip_gpu or not torch.cuda.is_available():
        prediction_report["gpu_forward"] = {
            "status": "skipped",
            "cuda_available": bool(torch.cuda.is_available()),
        }
        (out_dir / "prediction.json").write_text(json.dumps(prediction_report, indent=2) + "\n")
        print("GPU_FORWARD_SKIPPED")
        return

    # 2. Build model & strictly check checkpoint weight matching
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt_raw = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt_raw.get("state_dict", ckpt_raw)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model.CLASSES = ckpt_raw.get("meta", {}).get("CLASSES", val_dataset.CLASSES)
    model_dp = MMDataParallel(model.cuda().eval(), device_ids=[0])

    # 3. Run GPU forward with TraceRecorder hooks
    recorder = TraceRecorder()
    recorder.register(model_dp.module)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_orig = model_dp(return_loss=False, rescale=True, **copy.deepcopy(val_target_batch))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    recorder.close()

    # 4. Audit GT consumption in simple_test: repeat baseline (CUDA nondeterminism check), empty GT & perturbed GT
    with torch.inference_mode():
        output_orig_repeat = model_dp(
            return_loss=False,
            rescale=True,
            **copy.deepcopy(val_target_batch),
        )
        output_empty_gt = model_dp(
            return_loss=False,
            rescale=True,
            **_clone_batch_with_gt_mode(val_target_batch, "empty"),
        )
        output_perturbed_gt = model_dp(
            return_loss=False,
            rescale=True,
            **_clone_batch_with_gt_mode(val_target_batch, "perturbed"),
        )
    torch.cuda.synchronize()

    cmp_repeat = _compare_predictions(output_orig, output_orig_repeat)
    cmp_empty = _compare_predictions(output_orig, output_empty_gt)
    cmp_perturbed = _compare_predictions(output_orig, output_perturbed_gt)
    gt_independent = bool(cmp_empty["top10_close"] and cmp_perturbed["top10_close"])

    pred_summary = unwrap_prediction(output_orig)
    bev_path = out_dir / "bev_pred.png"
    render_v2x_bev(
        data_root=Path(cfg.data_root),
        frame_id=target_fid,
        dataset=val_dataset,
        sample_index=val_target_index,
        predictions=pred_summary["top10"],
        out_path=bev_path,
    )

    stage_names = [
        "image_fpn",
        "depth_distribution",
        "camera_bev",
        "radar_pillars",
        "radar_scatter",
        "radar_bev",
        "cross_modal_fusion",
        "temporal_fusion",
        "backward_projection",
        "instance_guided_refinement",
        "detection_head",
    ]
    stage_trace = {}
    for name in stage_names:
        calls = recorder.calls.get(name, [])
        stage_trace[name] = {
            "status": "PASS" if len(calls) > 0 else "SKIP",
            "call_count": len(calls),
            "calls": calls,
        }

    trace_path = out_dir / "stage_trace.json"
    trace_path.write_text(json.dumps(stage_trace, indent=2) + "\n", encoding="utf-8")

    prediction_report.update(
        {
            "status": "ok",
            "gpu_forward": {
                "status": "ok",
                "device": torch.cuda.get_device_name(0),
                "elapsed_seconds": round(elapsed, 4),
                "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
                "parameters": int(sum(p.numel() for p in model_dp.parameters())),
                "checkpoint_weight_match": {
                    "checkpoint_path": args.checkpoint,
                    "strict_match": len(incompatible.missing_keys) == 0
                    and len(incompatible.unexpected_keys) == 0,
                    "missing_keys": list(incompatible.missing_keys),
                    "unexpected_keys": list(incompatible.unexpected_keys),
                    "note": "TJ4D weights used strictly for Stage 2 cross-dataset forward verification.",
                },
                "predictions": pred_summary,
                "bev_visualization": str(bev_path),
                "stage_trace_path": str(trace_path),
                "stage_status_summary": {
                    k: {"status": v["status"], "call_count": v["call_count"]}
                    for k, v in stage_trace.items()
                },
            },
            "simple_test_gt_audit": {
                "code_locations": [
                    "mmdet3d/models/detectors/R4Det.py:L893-L918 (unconditionally unpacks gt_bboxes_3d/gt_labels_3d/gt_bboxes/gt_labels into img_metas)",
                    "mmdet3d/models/detectors/R4Det.py:L1003-L1006 (passes GT to draw_gt_pred_figures_3d every SAVE_INTERVALS=250 calls)",
                    "mmdet3d/models/detectors/R4Det.py:L1389-L1405 (extracts GT from img_metas in eval preprocessing_information for draw_pts_completion every 250 calls)",
                ],
                "original_vs_original_repeat": cmp_repeat,
                "original_vs_empty_gt": cmp_empty,
                "original_vs_perturbed_gt": cmp_perturbed,
                "predictions_depend_on_gt": not gt_independent,
                "unlabeled_inference_note": (
                    "R4Det.simple_test requires gt_bboxes_3d/gt_labels_3d/gt_bboxes/gt_labels keys "
                    "to be present (even if empty tensors) because L893-L918 index them unconditionally; "
                    "V2XRadarVDataset(allow_unlabeled=True) supplies valid empty GT tensors when annos=None. "
                    "Any ~1e-5 score jitter matches the identical-input repeat run (CUDA atomicAdd nondeterminism)."
                ),
            },
        }
    )

    pred_path = out_dir / "prediction.json"
    pred_path.write_text(json.dumps(prediction_report, indent=2) + "\n", encoding="utf-8")

    print(
        f"STAGE2_GPU_OK frame={target_fid} detections={pred_summary['detections']} "
        f"score_max={pred_summary['score_max']} "
        f"strict_ckpt={prediction_report['gpu_forward']['checkpoint_weight_match']['strict_match']} "
        f"gt_independent={not prediction_report['simple_test_gt_audit']['predictions_depend_on_gt']}"
    )
    print(f"  stage_status={prediction_report['gpu_forward']['stage_status_summary']}")
    print(f"  wrote: {pred_path}")
    print(f"  wrote: {trace_path}")
    print(f"  wrote: {bev_path}")


if __name__ == "__main__":
    main()
