"""Explicit standalone 3D/BEV AP evaluator for V2X-Radar-V.

Never overwrites ``mmdet3d/core/evaluation/kitti_utils/eval.py``.
Supports:
- ``v2x_benchmark``: official vehicle-side V2X-Radar-V 3-class protocol
  (Vehicle = Car/Van/Truck/Bus @ IoU 0.7 & 0.5; Pedestrian @ 0.5 & 0.25;
  Cyclist @ 0.5 & 0.25) with 40-recall-point AP (R40).
- ``tj4d_compat``: 4-class protocol (Pedestrian, Cyclist, Car, Truck).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from mmcv.ops import box_iou_rotated


V2X_BENCHMARK_THRESHOLDS: Dict[str, Tuple[float, ...]] = {
    "Vehicle": (0.7, 0.5),
    "Pedestrian": (0.5, 0.25),
    "Cyclist": (0.5, 0.25),
}

TJ4D_COMPAT_THRESHOLDS: Dict[str, Tuple[float, ...]] = {
    "Car": (0.5, 0.25),
    "Truck": (0.5, 0.25),
    "Pedestrian": (0.25,),
    "Cyclist": (0.25,),
}

V2X_CLASS_TO_BENCHMARK: Dict[str, str] = {
    "Car": "Vehicle",
    "Van": "Vehicle",
    "Truck": "Vehicle",
    "Bus": "Vehicle",
    "Vehicle": "Vehicle",
    "Pedestrian": "Pedestrian",
    "Cyclist": "Cyclist",
}


def compute_bev_and_3d_iou_lidar(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rotated BEV IoU and 3D IoU between LiDAR boxes [x, y, z_bot, dx, dy, dz, yaw]."""
    na, nb = boxes_a.shape[0], boxes_b.shape[0]
    if na == 0 or nb == 0:
        zeros = np.zeros((na, nb), dtype=np.float32)
        return zeros, zeros

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a_t = torch.from_numpy(boxes_a.astype(np.float32)).to(device)
    b_t = torch.from_numpy(boxes_b.astype(np.float32)).to(device)

    # mmcv box_iou_rotated expects [cx, cy, w, h, angle_rad]
    bev_a = a_t[:, [0, 1, 3, 4, 6]].contiguous()
    bev_b = b_t[:, [0, 1, 3, 4, 6]].contiguous()
    bev_iou = box_iou_rotated(bev_a, bev_b).detach().cpu().numpy()

    # Recover BEV intersection area from IoU: inter = iou * (area_a + area_b) / (1 + iou)
    area_a = (boxes_a[:, 3] * boxes_a[:, 4])[:, None]
    area_b = (boxes_b[:, 3] * boxes_b[:, 4])[None, :]
    bev_inter = bev_iou * (area_a + area_b) / np.maximum(1.0 + bev_iou, 1e-8)

    # Vertical overlap (LiDARInstance3DBoxes origin is bottom center: z in [z, z + dz])
    za_min = boxes_a[:, 2:3]
    za_max = za_min + boxes_a[:, 5:6]
    zb_min = boxes_b[:, 2][None, :]
    zb_max = zb_min + boxes_b[:, 5][None, :]
    z_inter = np.maximum(0.0, np.minimum(za_max, zb_max) - np.maximum(za_min, zb_min))

    inter_3d = bev_inter * z_inter
    vol_a = area_a * boxes_a[:, 5:6]
    vol_b = area_b * boxes_b[:, 5][None, :]
    iou_3d = inter_3d / np.maximum(vol_a + vol_b - inter_3d, 1e-8)
    return bev_iou.astype(np.float32), iou_3d.astype(np.float32)


def compute_ap_r40(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute KITTI / V2X-Radar 40-point interpolated Average Precision (R40)."""
    if recalls.size == 0 or precisions.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    recall_points = np.linspace(1.0 / 40.0, 1.0, 40)
    ap_sum = 0.0
    for r in recall_points:
        inds = np.where(mrec >= r)[0]
        ap_sum += float(mpre[inds[0]]) if inds.size > 0 else 0.0
    return (ap_sum / 40.0) * 100.0


def evaluate_v2x_radar_v(
    gt_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
    protocol: str = "v2x_benchmark",
    score_thr: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate 3D and BEV AP on V2X-Radar-V frames.

    Each element of ``gt_records`` must contain:
      - ``frame_id`` (str)
      - ``boxes_3d`` (np.ndarray, (M, 7) in LiDAR frame)
      - ``names`` (list[str])

    Each element of ``pred_records`` must contain:
      - ``frame_id`` (str)
      - ``boxes_3d`` (np.ndarray, (K, 7) in LiDAR frame)
      - ``scores_3d`` (np.ndarray, (K,))
      - ``names`` (list[str])
    """
    if protocol == "v2x_benchmark":
        thresholds_map = V2X_BENCHMARK_THRESHOLDS
        map_name = lambda n: V2X_CLASS_TO_BENCHMARK.get(n, None)
    elif protocol == "tj4d_compat":
        thresholds_map = TJ4D_COMPAT_THRESHOLDS
        map_name = lambda n: n if n in TJ4D_COMPAT_THRESHOLDS else None
    else:
        raise ValueError(f"Unsupported evaluation protocol: {protocol}")

    assert len(gt_records) == len(pred_records), (
        f"Mismatch between GT ({len(gt_records)}) and pred ({len(pred_records)}) counts."
    )

    metrics: Dict[str, Any] = {
        "protocol": protocol,
        "num_frames": len(gt_records),
        "classes": {},
    }

    for cls_name, iou_thrs in thresholds_map.items():
        cls_result: Dict[str, Any] = {"iou_thresholds": list(iou_thrs)}

        # Gather all GT and predictions for this class across frames
        gt_by_frame: Dict[int, np.ndarray] = {}
        total_gt = 0
        all_preds: List[Tuple[float, int, np.ndarray]] = []

        for f_idx, (gt_rec, pred_rec) in enumerate(zip(gt_records, pred_records)):
            gt_mask = [i for i, n in enumerate(gt_rec["names"]) if map_name(str(n)) == cls_name]
            gt_boxes = (
                np.asarray(gt_rec["boxes_3d"], dtype=np.float32)[gt_mask]
                if gt_mask
                else np.zeros((0, 7), dtype=np.float32)
            )
            gt_by_frame[f_idx] = gt_boxes
            total_gt += int(gt_boxes.shape[0])

            p_boxes = np.asarray(pred_rec["boxes_3d"], dtype=np.float32)
            p_scores = np.asarray(pred_rec["scores_3d"], dtype=np.float32)
            for i, n in enumerate(pred_rec["names"]):
                if map_name(str(n)) == cls_name and float(p_scores[i]) >= score_thr:
                    all_preds.append((float(p_scores[i]), f_idx, p_boxes[i]))

        all_preds.sort(key=lambda x: x[0], reverse=True)
        cls_result["num_gt"] = total_gt
        cls_result["num_pred"] = len(all_preds)

        for mode in ("bev", "3d"):
            cls_result[f"ap_{mode}_r40"] = {}
            for thr in iou_thrs:
                if total_gt == 0 or len(all_preds) == 0:
                    cls_result[f"ap_{mode}_r40"][f"iou_{thr}"] = 0.0
                    continue

                matched = {f_idx: np.zeros((gt_by_frame[f_idx].shape[0],), dtype=bool) for f_idx in gt_by_frame}
                tp = np.zeros((len(all_preds),), dtype=np.float32)
                fp = np.zeros((len(all_preds),), dtype=np.float32)

                for p_i, (_score, f_idx, p_box) in enumerate(all_preds):
                    gts = gt_by_frame[f_idx]
                    if gts.shape[0] == 0:
                        fp[p_i] = 1.0
                        continue
                    bev_iou, iou_3d = compute_bev_and_3d_iou_lidar(p_box[None, :], gts)
                    ious = bev_iou[0] if mode == "bev" else iou_3d[0]
                    best_idx = int(np.argmax(ious))
                    best_iou = float(ious[best_idx])
                    if best_iou >= thr and not matched[f_idx][best_idx]:
                        tp[p_i] = 1.0
                        matched[f_idx][best_idx] = True
                    else:
                        fp[p_i] = 1.0

                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(fp)
                recalls = cum_tp / float(total_gt)
                precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
                ap = compute_ap_r40(recalls, precisions)
                cls_result[f"ap_{mode}_r40"][f"iou_{thr}"] = round(ap, 4)

        metrics["classes"][cls_name] = cls_result

    # Compute mean AP at primary (strict) and secondary (relaxed) thresholds
    primary_3d = []
    primary_bev = []
    for cls_name, iou_thrs in thresholds_map.items():
        if metrics["classes"][cls_name]["num_gt"] > 0:
            t0 = f"iou_{iou_thrs[0]}"
            primary_3d.append(metrics["classes"][cls_name]["ap_3d_r40"][t0])
            primary_bev.append(metrics["classes"][cls_name]["ap_bev_r40"][t0])
    metrics["mAP_3d_primary_r40"] = round(float(np.mean(primary_3d)), 4) if primary_3d else 0.0
    metrics["mAP_bev_primary_r40"] = round(float(np.mean(primary_bev)), 4) if primary_bev else 0.0
    return metrics
