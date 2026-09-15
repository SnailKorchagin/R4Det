#!/usr/bin/env python3
"""Run one R4Det TJ4D sample through the full GPU inference path."""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/r4det/TJ4D-R4Det_det3d_2x4_12e.py"
    )
    parser.add_argument("--checkpoint", default="checkpoints/final_tj4d.pth")
    parser.add_argument(
        "--ann-file", default="data/TJ4D/TJ4D_infos_public_smoke.pkl"
    )
    parser.add_argument("--output", default=".smoke/r4det_result.json")
    parser.add_argument("--bev-output", default=".smoke/r4det_bev.png")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def unwrap_prediction(output):
    prediction = output[0]
    if "pts_bbox" in prediction:
        prediction = prediction["pts_bbox"]
    boxes = prediction["boxes_3d"].tensor.detach().cpu()
    scores = prediction["scores_3d"].detach().cpu()
    labels = prediction["labels_3d"].detach().cpu()
    order = torch.argsort(scores, descending=True)[:10]
    return {
        "detections": int(scores.numel()),
        "score_max": float(scores.max()) if scores.numel() else None,
        "top10": [
            {
                "score": float(scores[i]),
                "label": int(labels[i]),
                "box": [round(float(value), 5) for value in boxes[i].tolist()],
            }
            for i in order
        ],
    }


def describe_batch_value(value):
    """Return compact shapes without assuming an MMCV container layout."""
    if hasattr(value, "tensor"):
        return list(value.tensor.shape)
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [describe_batch_value(item) for item in value]
    return type(value).__name__


def render_bev(dataset, predictions, output_path):
    """Render real radar points, ground truth and top predictions in BEV."""
    height, width = 900, 900
    x_min, x_max = 0.0, 69.12
    y_min, y_max = -39.68, 39.68
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)

    def pixel(xy):
        x, y = xy
        col = int((y_max - y) / (y_max - y_min) * (width - 1))
        row = int((x_max - x) / (x_max - x_min) * (height - 1))
        return col, row

    info = dataset.data_infos[0]
    frame_id = f"{int(info['image']['image_idx']):06d}"
    radar_path = Path(dataset.data_root) / "training" / "velodyne_reduced" / f"{frame_id}.bin"
    radar = np.fromfile(radar_path, dtype=np.float32).reshape(-1, 8)
    valid = (
        (radar[:, 0] >= x_min)
        & (radar[:, 0] <= x_max)
        & (radar[:, 1] >= y_min)
        & (radar[:, 1] <= y_max)
    )
    for x, y in radar[valid, :2]:
        cv2.circle(canvas, pixel((x, y)), 1, (155, 155, 155), -1)

    def draw_box(box, color, text):
        x, y, _, dx, dy, _, yaw = box[:7]
        local = np.array(
            [[dx / 2, dy / 2], [dx / 2, -dy / 2], [-dx / 2, -dy / 2], [-dx / 2, dy / 2]],
            dtype=np.float32,
        )
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=np.float32,
        )
        corners = local @ rotation.T + np.array([x, y], dtype=np.float32)
        polygon = np.array([pixel(point) for point in corners], dtype=np.int32)
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, text, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    gt = dataset.get_ann_info(0)
    gt_boxes = gt["gt_bboxes_3d"].tensor.cpu().numpy()
    for box, label in zip(gt_boxes, gt["gt_labels_3d"]):
        name = dataset.CLASSES[int(label)] if label >= 0 else "ignore"
        draw_box(box, (30, 70, 255), f"GT {name}")
    for prediction in predictions:
        label = int(prediction["label"])
        name = dataset.CLASSES[label]
        draw_box(prediction["box"], (70, 230, 70), f"P {name} {prediction['score']:.3f}")

    cv2.putText(canvas, "R4Det public smoke: radar BEV", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
    cv2.putText(canvas, "gray=radar  red=GT  green=prediction", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to write {output_path}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    logging.getLogger("mmcv").setLevel(logging.WARNING)
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    figure_dir = Path(".smoke/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    cfg.model.update(
        meta_info={
            "figures_path": str(figure_dir),
            "project_name": "TJ4D-R4Det-public-smoke",
        }
    )
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = False

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    data = next(iter(loader))
    points = describe_batch_value(data["points"].data)
    image = describe_batch_value(data["img"].data)
    print(f"PIPELINE_OK points={points} image={image}")

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    if args.fp16:
        wrap_fp16_model(model)
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(return_loss=False, rescale=True, **data)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    summary = unwrap_prediction(output)
    render_bev(dataset, summary["top10"], args.bev_output)
    summary.update(
        {
            "status": "ok",
            "device": torch.cuda.get_device_name(0),
            "fp16": bool(args.fp16),
            "elapsed_seconds": round(elapsed, 4),
            "peak_gpu_memory_mib": round(
                torch.cuda.max_memory_allocated() / (1024**2), 1
            ),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "input_is_original_camera": False,
            "bev_visualization": args.bev_output,
        }
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("R4DET_SMOKE_OK")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
