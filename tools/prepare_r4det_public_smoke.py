#!/usr/bin/env python3
"""Prepare one public TJ4D radar frame for an R4Det code-path smoke test.

The public TJ4DRadSet sample contains radar, calibration and labels, but not
the synchronized camera image. This tool therefore creates a radar-projection
proxy image. It is useful for validating the complete software path, but its
predictions are not a reproduction of the paper's reported accuracy.
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/TJ4D")
    parser.add_argument("--frame-id", default="070070")
    return parser.parse_args()


def load_frame_info(path, frame_id):
    with path.open("rb") as handle:
        infos = pickle.load(handle)
    for info in infos:
        if f"{int(info['image']['image_idx']):06d}" == frame_id:
            return info
    raise RuntimeError(f"frame {frame_id} is absent from {path}")


def project_radar(info, radar, height, width):
    lidar2img = (
        info["calib"]["P2"].astype(np.float32)
        @ info["calib"]["R0_rect"].astype(np.float32)
        @ info["calib"]["Tr_velo_to_cam"].astype(np.float32)
    )
    xyz1 = np.concatenate(
        [radar[:, :3], np.ones((radar.shape[0], 1), dtype=np.float32)], axis=1
    )
    projected = xyz1 @ lidar2img.T
    depth = projected[:, 2]
    uv = projected[:, :2] / np.maximum(depth[:, None], 1e-6)
    valid = (
        (depth > 1.0)
        & (depth < 80.0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv[valid], depth[valid], valid


def make_proxy_image(uv, depth, radar_strength, height, width):
    # A dim background keeps the visual tensor well-defined while ensuring
    # every visible structure comes from the real, calibrated radar frame.
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(28, 3, height, dtype=np.uint8)[:, None]
    image[:, :, 1] = np.linspace(20, 3, height, dtype=np.uint8)[:, None]
    image[:, :, 2] = np.linspace(12, 2, height, dtype=np.uint8)[:, None]
    mask = np.zeros((height, width), dtype=np.uint8)

    if len(depth):
        strength = radar_strength.astype(np.float32)
        strength = (strength - np.percentile(strength, 5)) / max(
            np.percentile(strength, 95) - np.percentile(strength, 5), 1e-6
        )
        strength = np.clip(strength, 0.0, 1.0)
        for (u, v), distance, power in zip(uv, depth, strength):
            color = cv2.applyColorMap(
                np.array([[int(255 * (1.0 - distance / 80.0))]], dtype=np.uint8),
                cv2.COLORMAP_TURBO,
            )[0, 0]
            radius = 2 if power > 0.5 else 1
            cv2.circle(image, (int(u), int(v)), radius, color.tolist(), -1)
            cv2.circle(mask, (int(u), int(v)), 2, 1, -1)
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return image, mask.astype(np.float32)


def main():
    args = parse_args()
    root = Path(args.data_root)
    frame_id = args.frame_id
    info = load_frame_info(root / "TJ4D_infos_train.pkl", frame_id)

    radar_path = root / "training" / "velodyne_reduced" / f"{frame_id}.bin"
    radar = np.fromfile(radar_path, dtype=np.float32)
    if radar.size % 8:
        raise RuntimeError(f"invalid 8-float radar record file: {radar_path}")
    radar = radar.reshape(-1, 8)

    height, width = map(int, info["image"]["image_shape"])
    uv, depth, valid = project_radar(info, radar, height, width)
    image, segmentation = make_proxy_image(
        uv, depth, radar[valid, 5], height, width
    )

    image_path = root / "training" / "image_2" / f"{frame_id}.png"
    segmentation_path = root / "segmentation" / f"{frame_id}.npy"
    smoke_info_path = root / "TJ4D_infos_public_smoke.pkl"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    segmentation_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"failed to write {image_path}")
    np.save(segmentation_path, segmentation)
    with smoke_info_path.open("wb") as handle:
        pickle.dump([info], handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "frame_id": frame_id,
        "radar_source": "TJRadarLab/TJ4DRadSet public Sample/training/velodyne",
        "radar_points": int(radar.shape[0]),
        "radar_points_projected_into_proxy": int(valid.sum()),
        "calibration_and_labels_source": "Hoeyy/R4Det TJ4D_infos_train.pkl",
        "camera_input": "radar_projection_proxy_not_original_camera",
        "valid_for_paper_metric_reproduction": False,
        "image_shape": [height, width],
        "ground_truth_objects": int(len(info["annos"]["name"])),
    }
    manifest_path = root / "public_smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"prepared {smoke_info_path}")


if __name__ == "__main__":
    main()
