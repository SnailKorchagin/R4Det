"""V2X-Radar-V loading and optional auxiliary transforms for R4Det."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES

from mmdet3d.core.points import LiDARPoints
from mmdet3d.datasets.v2x_radar_v_utils import (
    format_radar_points_for_r4det,
    read_radar_bin,
)


@PIPELINES.register_module()
class LoadV2XRadarPoints:
    """Load raw V2X-Radar-V 5-channel 4D radar points and convert to R4Det 5D.

    Raw file schema: ``[x_radar, y_radar, z_radar, intensity, velocity]``.
    Output schema in Velodyne/LiDAR coordinates:
    ``[x_velo, y_velo, z_velo, velocity, (intensity - intensity_offset) * intensity_scale]``.
    """

    def __init__(
        self,
        num_features: int = 5,
        intensity_offset: float = 0.0,
        intensity_scale: float = 1.0,
    ):
        self.num_features = int(num_features)
        self.intensity_offset = float(intensity_offset)
        self.intensity_scale = float(intensity_scale)

    def __call__(self, results: dict) -> dict:
        pts_filename = results["pts_filename"]
        if not os.path.isfile(pts_filename):
            raise FileNotFoundError(f"Missing V2X-Radar-V radar file: {pts_filename}")

        calib = results.get("calib", {})
        if "Tr_radar_to_velo" not in calib:
            raise KeyError("Missing 'Tr_radar_to_velo' in results['calib']")
        tr_radar_to_velo = np.asarray(calib["Tr_radar_to_velo"], dtype=np.float64)

        raw_pts = read_radar_bin(pts_filename, num_features=self.num_features)
        r4det_pts = format_radar_points_for_r4det(
            raw_pts,
            tr_radar_to_velo=tr_radar_to_velo,
            intensity_offset=self.intensity_offset,
            intensity_scale=self.intensity_scale,
        )

        results["points"] = LiDARPoints(
            r4det_pts, points_dim=5, attribute_dims=None
        )
        results["radar_intensity_cfg"] = dict(
            offset=self.intensity_offset,
            scale=self.intensity_scale,
            raw_column=3,
            r4det_column=4,
        )
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_features={self.num_features}, "
            f"intensity_offset={self.intensity_offset}, "
            f"intensity_scale={self.intensity_scale})"
        )


@PIPELINES.register_module()
class LoadOptionalSegmentation:
    """Load precomputed 2D segmentation mask if present, else fallback to zeros."""

    def __init__(
        self,
        data_root: Optional[str] = None,
        seg_subdir: str = "segmentation",
        allow_missing: bool = True,
    ):
        self.data_root = data_root
        self.seg_subdir = seg_subdir
        self.allow_missing = allow_missing

    def __call__(self, results: dict) -> dict:
        frame_id = results.get("frame_id")
        if frame_id is None:
            frame_id = Path(results["pts_filename"]).stem

        seg_path = None
        if self.data_root is not None:
            candidate = Path(self.data_root) / self.seg_subdir / f"{frame_id}.npy"
            if candidate.is_file():
                seg_path = candidate

        if seg_path is not None:
            seg = np.load(str(seg_path)).astype(np.bool_)
            results["segmentation"] = seg
            results["segmentation_source"] = str(seg_path)
            return results

        if not self.allow_missing:
            raise FileNotFoundError(
                f"Missing segmentation mask for frame {frame_id} under "
                f"{self.data_root}/{self.seg_subdir}"
            )

        H, W = results["ori_shape"][:2]
        results["segmentation"] = np.zeros((H, W), dtype=np.bool_)
        results["segmentation_source"] = "zeros_fallback"
        return results


@PIPELINES.register_module()
class GenerateBoxMasksFrom2D:
    """Generate reproducible BitmapMasks from post-augmentation 2D GT boxes.

    Satisfies ``R4Det.img_roi_head._mask_forward_train`` without requiring
    offline VLSAM/Detectron2 mask files. If an offline ``gt_masks/{fid}.npy``
    file exists under ``data_root`` with matching count, it can be loaded;
    otherwise each valid 2D box ``[x1, y1, x2, y2]`` is rasterized as a binary
    foreground rectangle on the ``(H, W)`` augmented image grid.
    """

    def __init__(self, data_root: Optional[str] = None, mask_subdir: str = "gt_masks"):
        self.data_root = data_root
        self.mask_subdir = mask_subdir

    def __call__(self, results: dict) -> dict:
        H, W = results["img_shape"][:2]
        gt_bboxes = results.get("gt_bboxes", None)
        if gt_bboxes is None:
            bboxes_np = np.zeros((0, 4), dtype=np.float32)
        elif torch.is_tensor(gt_bboxes):
            bboxes_np = gt_bboxes.detach().cpu().numpy().reshape(-1, 4)
        else:
            bboxes_np = np.asarray(gt_bboxes, dtype=np.float32).reshape(-1, 4)

        num_boxes = bboxes_np.shape[0]
        if num_boxes == 0:
            masks = np.zeros((0, H, W), dtype=np.uint8)
        else:
            masks = np.zeros((num_boxes, H, W), dtype=np.uint8)
            for i, box in enumerate(bboxes_np):
                x1 = int(np.clip(np.floor(box[0]), 0, W - 1))
                y1 = int(np.clip(np.floor(box[1]), 0, H - 1))
                x2 = int(np.clip(np.ceil(box[2]), x1 + 1, W))
                y2 = int(np.clip(np.ceil(box[3]), y1 + 1, H))
                masks[i, y1:y2, x1:x2] = 1

        results["gt_masks"] = BitmapMasks(masks, H, W)
        results["gt_masks_source"] = "box2d_rasterized"
        return results


@PIPELINES.register_module()
class CreateOptionalDepthFromLiDAR:
    """Project V2X-Radar-V LiDAR points to gt_depths (and my_gt_depth).

    Priority:
      1. Real V2X-Radar-V LiDAR ``training/velodyne/{frame_id}.bin`` (4-col float32).
      2. Sparse fallback from ``radar_depth`` (if LiDAR file is absent locally) so
         ``GeometryDepth_Net.get_depth_loss`` never divides by zero / returns NaN.
    Also produces ``my_gt_depth`` (numpy float32 of ``my_depth_shape``) required
    by ``R4Det.forward_train``.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        lidar_subdir: str = "training/velodyne",
        min_depth: float = 0.0,
        max_depth: float = 80.0,
        allow_missing: bool = True,
        fallback_to_radar: bool = True,
        my_depth_shape: Tuple[int, int] = (60, 80),
    ):
        self.data_root = data_root
        self.lidar_subdir = lidar_subdir
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.allow_missing = allow_missing
        self.fallback_to_radar = fallback_to_radar
        self.my_depth_shape = (int(my_depth_shape[0]), int(my_depth_shape[1]))

    def _make_my_gt_depth(self, gt_depth_np: np.ndarray) -> np.ndarray:
        th, tw = self.my_depth_shape
        return cv2.resize(
            gt_depth_np.astype(np.float32),
            dsize=(tw, th),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

    def __call__(self, results: dict) -> dict:
        H, W = results["img_shape"][:2]
        frame_id = results.get("frame_id") or Path(results["pts_filename"]).stem

        lidar_path = results.get("lidar_filename")
        if lidar_path is None and self.data_root is not None:
            candidate = Path(self.data_root) / self.lidar_subdir / f"{frame_id}.bin"
            if candidate.is_file():
                lidar_path = str(candidate)

        if lidar_path is None or not os.path.isfile(lidar_path):
            if not self.allow_missing:
                raise FileNotFoundError(
                    f"Missing LiDAR file for depth supervision: frame {frame_id}"
                )
            if self.fallback_to_radar and "radar_depth" in results:
                gt_depth = np.asarray(results["radar_depth"], dtype=np.float32).copy()
                results["gt_depths"] = torch.from_numpy(gt_depth)
                results["gt_depths_source"] = "radar_depth_fallback"
                results["my_gt_depth"] = self._make_my_gt_depth(gt_depth)
                return results

            gt_depth = np.zeros((H, W), dtype=np.float32)
            results["gt_depths"] = torch.from_numpy(gt_depth)
            results["gt_depths_source"] = "zeros_fallback"
            results["my_gt_depth"] = self._make_my_gt_depth(gt_depth)
            return results

        lidar_raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        cam2img = deepcopy(results["cam2img"]).astype(np.float64)
        rots, trans, intrins, post_rots, post_trans = results["cam_aware"][:5]
        post_rots_np = (
            post_rots.detach().cpu().numpy()
            if torch.is_tensor(post_rots)
            else np.asarray(post_rots)
        )
        post_trans_np = (
            post_trans.detach().cpu().numpy()
            if torch.is_tensor(post_trans)
            else np.asarray(post_trans)
        )
        cam2img[:2, :3] = post_rots_np[:2, :2] @ cam2img[:2, :3]
        cam2img[:2, 2] = post_trans_np[:2] + cam2img[:2, 2]
        lidar2img = cam2img @ np.asarray(results["lidar2cam"], dtype=np.float64)

        pts_hom = np.concatenate([lidar_raw, np.ones((lidar_raw.shape[0], 1))], axis=1)
        proj = (lidar2img @ pts_hom.T).T
        depth = proj[:, 2]
        valid = depth > max(self.min_depth, 1e-6)
        proj = proj[valid]
        depth = depth[valid]
        u = proj[:, 0] / depth
        v = proj[:, 1] / depth
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth < self.max_depth)

        gt_depth = np.zeros((H, W), dtype=np.float32)
        if np.any(in_img):
            u_i = np.floor(u[in_img]).astype(np.int64)
            v_i = np.floor(v[in_img]).astype(np.int64)
            d_i = depth[in_img].astype(np.float32)
            order = np.argsort(-d_i)
            gt_depth[v_i[order], u_i[order]] = d_i[order]

        results["gt_depths"] = torch.from_numpy(gt_depth)
        results["gt_depths_source"] = str(lidar_path)
        results["my_gt_depth"] = self._make_my_gt_depth(gt_depth)
        return results
