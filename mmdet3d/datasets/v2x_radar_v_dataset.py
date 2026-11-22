"""MMDet3D / R4Det Dataset implementation for V2X-Radar-V (vehicle-side)."""

from __future__ import annotations

import copy
import os
from os import path as osp
from typing import Dict, Optional

import numpy as np
from mmdet.datasets import DATASETS

from mmdet3d.core.bbox import CameraInstance3DBoxes
from .TJ4D_dataset import TJ4DDataset
from .v2x_radar_v_utils import (
    TJ4D_COMPAT_CLASS_MAPPING,
    TJ4D_COMPAT_CLASSES,
    normalize_frame_id,
)


@DATASETS.register_module()
class V2XRadarVDataset(TJ4DDataset):
    """V2X-Radar-V dataset adapter for R4Det.

    Reuses ``TJ4DDataset``'s temporal packing and KITTI result formatting while
    using V2X-Radar-V's 6-digit frame IDs, ``Tr_radar_to_velo`` calibration,
    explicit class mapping, empty-GT / unlabeled handling, and explicit
    single-frame ``use_temporal=False`` control.
    """

    CLASSES = TJ4D_COMPAT_CLASSES

    def __init__(
        self,
        data_root,
        ann_file,
        split="training",
        pts_prefix="radar",
        pipeline=None,
        classes=None,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        pcd_limit_range=[0, -40, -3, 70.4, 40, 2.0],
        class_mapping: Optional[Dict[str, Optional[str]]] = None,
        use_temporal: bool = False,
        allow_unlabeled: bool = False,
    ):
        self.class_mapping = (
            dict(class_mapping)
            if class_mapping is not None
            else dict(TJ4D_COMPAT_CLASS_MAPPING)
        )
        self.use_temporal = bool(use_temporal)
        self.allow_unlabeled = bool(allow_unlabeled)
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range,
        )

    def _get_pts_filename(self, idx, info=None):
        if info is not None:
            pc_info = info.get("point_cloud", {})
            rel_path = pc_info.get("radar_path") or pc_info.get("velodyne_path")
            if rel_path:
                return osp.join(self.data_root, rel_path)
        fid = normalize_frame_id(idx)
        return osp.join(self.root_split, self.pts_prefix, f"{fid}.bin")

    def get_data_info(self, index):
        info = self.data_infos[index]
        sample_idx = info["image"]["image_idx"]
        frame_id = info["image"].get("frame_id") or normalize_frame_id(sample_idx)
        img_filename = osp.join(self.data_root, info["image"]["image_path"])
        pts_filename = self._get_pts_filename(sample_idx, info=info)

        if not osp.isfile(pts_filename):
            raise FileNotFoundError(
                f"Missing critical radar file for frame {frame_id}: {pts_filename}"
            )
        if not osp.isfile(img_filename):
            raise FileNotFoundError(
                f"Missing critical image file for frame {frame_id}: {img_filename}"
            )

        calib = info["calib"]
        for req_key in ("P2", "R0_rect", "Tr_velo_to_cam", "Tr_radar_to_velo"):
            if req_key not in calib:
                raise KeyError(
                    f"Missing required calibration matrix '{req_key}' for frame {frame_id}"
                )

        rect = np.asarray(calib["R0_rect"], dtype=np.float32)
        tr_v2c = np.asarray(calib["Tr_velo_to_cam"], dtype=np.float32)
        p2 = np.asarray(calib["P2"], dtype=np.float32)

        lidar2cam = rect @ tr_v2c
        lidar2img = p2 @ lidar2cam
        cam2img = p2
        focal_length = float(cam2img[0, 0])
        baseline = float(self.dynamic_baseline(info))

        lidar_rel = info.get("point_cloud", {}).get("lidar_path")
        lidar_filename = osp.join(self.data_root, lidar_rel) if lidar_rel else None

        input_dict = dict(
            sample_idx=sample_idx,
            frame_id=frame_id,
            pts_filename=pts_filename,
            lidar_filename=lidar_filename,
            img_prefix=None,
            img_info=dict(filename=img_filename),
            lidar2img=lidar2img,
            lidar2cam=lidar2cam,
            cam2img=cam2img,
            focal_length=focal_length,
            baseline=baseline,
            calib=calib,
            has_annotations=(info.get("annos") is not None),
        )

        if info.get("annos") is not None:
            input_dict["ann_info"] = self.get_ann_info(index)
        elif self.allow_unlabeled or self.test_mode:
            input_dict["ann_info"] = self._empty_ann_info(rect, tr_v2c)
        else:
            raise ValueError(
                f"Frame {frame_id} has no annotations (annos=None) while "
                "allow_unlabeled=False and test_mode=False."
            )

        return input_dict

    def _empty_ann_info(self, rect: np.ndarray, tr_v2c: np.ndarray) -> dict:
        empty_cam = CameraInstance3DBoxes(np.zeros((0, 7), dtype=np.float32))
        empty_boxes = empty_cam.convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ tr_v2c)
        )
        return dict(
            gt_bboxes_3d=empty_boxes,
            gt_labels_3d=np.zeros((0,), dtype=np.int64),
            bboxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            gt_names=np.array([], dtype="<U32"),
            raw_names=np.array([], dtype="<U32"),
        )

    def _map_class_name(self, raw_name: str) -> Optional[str]:
        if raw_name in self.class_mapping:
            return self.class_mapping[raw_name]
        return raw_name

    def get_ann_info(self, index):
        info = self.data_infos[index]
        rect = np.asarray(info["calib"]["R0_rect"], dtype=np.float32)
        tr_v2c = np.asarray(info["calib"]["Tr_velo_to_cam"], dtype=np.float32)

        annos = info.get("annos")
        if annos is None:
            if self.allow_unlabeled or self.test_mode:
                return self._empty_ann_info(rect, tr_v2c)
            raise ValueError(f"Missing annos for index {index}")

        # Handle legitimate empty-GT frame cleanly
        if len(annos.get("name", [])) == 0:
            return self._empty_ann_info(rect, tr_v2c)

        raw_names_all = annos.get("raw_name", annos["name"])
        mapped_names_all = np.array(
            [
                "DontCare" if self._map_class_name(str(n)) is None else self._map_class_name(str(n))
                for n in raw_names_all
            ],
            dtype="<U32",
        )

        keep_mask = mapped_names_all != "DontCare"
        if not np.any(keep_mask):
            return self._empty_ann_info(rect, tr_v2c)

        loc = np.asarray(annos["location"], dtype=np.float32)[keep_mask].reshape(-1, 3)
        dims = np.asarray(annos["dimensions"], dtype=np.float32)[keep_mask].reshape(-1, 3)
        rots = np.asarray(annos["rotation_y"], dtype=np.float32)[keep_mask].reshape(-1)
        gt_bboxes = np.asarray(annos["bbox"], dtype=np.float32)[keep_mask].reshape(-1, 4)
        gt_names = mapped_names_all[keep_mask]
        raw_names = np.asarray(raw_names_all, dtype="<U32")[keep_mask]

        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(
            np.float32
        )
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ tr_v2c)
        )

        gt_labels = np.array(
            [self.CLASSES.index(cat) if cat in self.CLASSES else -1 for cat in gt_names],
            dtype=np.int64,
        )
        gt_labels_3d = copy.deepcopy(gt_labels)

        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names,
            raw_names=raw_names,
        )

    def __getitem__(self, idx):
        """Return (prev, curr) pair expected by R4Det.

        When ``use_temporal=False`` (default for V2X-Radar-V), ``prev`` is always
        an empty frame with ``is_prev_frame_valid=False``.
        """
        while True:
            has_prev_frame = False
            if self.use_temporal and idx > 0:
                info_curr = self.data_infos[idx]
                info_prev = self.data_infos[idx - 1]
                curr_id = info_curr["image"]["image_idx"]
                prev_id = info_prev["image"]["image_idx"]
                if isinstance(curr_id, int) and isinstance(prev_id, int) and (curr_id - prev_id == 1):
                    has_prev_frame = True

            prepare_func = self.prepare_test_data if self.test_mode else self.prepare_train_data
            data_curr = prepare_func(idx)
            if data_curr is None:
                if self.test_mode:
                    return None
                idx = self._rand_another(idx)
                continue

            if has_prev_frame:
                data_prev = prepare_func(idx - 1)
                if data_prev is None:
                    has_prev_frame = False
            if not has_prev_frame:
                data_prev = self._create_empty_prev_data(data_curr)

            return self._pack_final_data(data_prev, data_curr, has_prev_frame)
