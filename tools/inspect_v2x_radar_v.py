#!/usr/bin/env python3
"""Stage 1 CPU audit and radar-on-image projection CLI for V2X-Radar-V.

Uses only Python stdlib, NumPy, Pillow, and OpenCV/Matplotlib (no CUDA import)
while sharing the exact parsing and geometry code in
``mmdet3d/datasets/v2x_radar_v_utils.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_utils_module():
    """Load v2x_radar_v_utils directly to avoid triggering mmdet3d/CUDA imports."""
    utils_path = REPO_ROOT / "mmdet3d" / "datasets" / "v2x_radar_v_utils.py"
    spec = importlib.util.spec_from_file_location("v2x_radar_v_utils", str(utils_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UTILS = _load_utils_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V2X-Radar-V dataset layout, splits, calibration, and radar projection (CPU)."
    )
    parser.add_argument(
        "--data-root",
        default="data/V2X-Radar-V",
        help="Path to V2X-Radar-V dataset root (containing training/ and ImageSets/).",
    )
    parser.add_argument(
        "--frame-id",
        required=True,
        help="Frame ID (e.g. 000009) to inspect and render radar_on_image.png.",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/v2x_radar_v/inspection",
        help="Output directory for inspection.json and radar_on_image.png.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=1e-3,
        help="Minimum positive camera depth (meters) for valid projection.",
    )
    return parser.parse_args()


def run_geometry_self_checks(calib: Dict[str, np.ndarray], img_shape: Tuple[int, int]) -> Dict[str, Any]:
    """Run deterministic geometry checks on known synthetic points."""
    H, W = img_shape
    P2 = calib["P2"]
    R0 = calib["R0_rect"]
    lidar2cam = calib["lidar2cam"]
    radar2velo = calib["radar2velo"]
    radar2cam = calib["radar2cam"]
    radar2img = calib["radar2img"]

    # 1. Known synthetic point in camera coordinates (in front of camera, on principal axis)
    fx, fy = float(P2[0, 0]), float(P2[1, 1])
    cx, cy = float(P2[0, 2]), float(P2[1, 2])
    tx, ty, tz = float(P2[0, 3]), float(P2[1, 3]), float(P2[2, 3])
    known_cam = np.array([2.0, 0.5, 20.0], dtype=np.float64)
    expected_u = (fx * known_cam[0] + cx * known_cam[2] + tx) / (known_cam[2] + tz)
    expected_v = (fy * known_cam[1] + cy * known_cam[2] + ty) / (known_cam[2] + tz)

    # Map known_cam back to radar coordinates, then forward-project through radar2img
    cam2radar = np.linalg.inv(radar2cam)
    known_radar = UTILS.transform_points_homogeneous(known_cam[None, :], cam2radar)
    known_proj = UTILS.project_points_to_image(known_radar, radar2img, img_shape)
    known_ok = (
        known_proj["in_image_count"] == 1
        and abs(float(known_proj["uv"][0, 0]) - expected_u) < 1e-4
        and abs(float(known_proj["uv"][0, 1]) - expected_v) < 1e-4
        and abs(float(known_proj["depth"][0]) - known_cam[2]) < 1e-4
    )

    # 2. Negative depth (behind camera), out-of-image, and NaN/Inf filtering
    behind_cam = np.array([0.0, 0.0, -5.0], dtype=np.float64)
    out_left_cam = np.array([-50.0, 0.0, 10.0], dtype=np.float64)
    out_top_cam = np.array([0.0, -50.0, 10.0], dtype=np.float64)
    test_cam = np.vstack([known_cam, behind_cam, out_left_cam, out_top_cam])
    test_radar = UTILS.transform_points_homogeneous(test_cam, cam2radar)
    test_radar = np.vstack([test_radar, np.array([[np.nan, 1.0, 1.0], [np.inf, 1.0, 1.0]])])
    filter_res = UTILS.project_points_to_image(test_radar, radar2img, img_shape)
    filter_ok = (
        filter_res["raw_count"] == 6
        and filter_res["finite_count"] == 4
        and filter_res["in_front_count"] == 3
        and filter_res["in_image_count"] == 1
        and bool(filter_res["valid_mask"][0]) is True
        and bool(filter_res["valid_mask"][1:].any()) is False
    )

    # 3. Composition consistency: P2 @ R0 @ Tr_velo_to_cam @ Tr_radar_to_velo == radar2img
    recomposed = P2 @ R0 @ calib["Tr_velo_to_cam"] @ radar2velo
    max_comp_err = float(np.max(np.abs(recomposed - radar2img)))
    comp_ok = max_comp_err < 1e-9

    # Also verify two-stage transform_radar_to_velo + lidar2img matches radar2img
    velo_pt = UTILS.transform_points_homogeneous(known_radar, radar2velo)
    two_stage = UTILS.project_points_to_image(velo_pt, calib["lidar2img"], img_shape)
    two_stage_err = float(np.max(np.abs(two_stage["uv"] - known_proj["uv"])))
    two_stage_ok = two_stage_err < 1e-5

    return {
        "known_point_projection": {
            "passed": bool(known_ok),
            "known_cam_xyz": known_cam.tolist(),
            "expected_uv": [round(expected_u, 4), round(expected_v, 4)],
            "actual_uv": [round(float(known_proj["uv"][0, 0]), 4), round(float(known_proj["uv"][0, 1]), 4)]
            if known_proj["in_image_count"] == 1
            else None,
            "actual_depth": round(float(known_proj["depth"][0]), 4)
            if known_proj["in_image_count"] == 1
            else None,
        },
        "filtering_negative_depth_oob_nonfinite": {
            "passed": bool(filter_ok),
            "raw_count": filter_res["raw_count"],
            "finite_count": filter_res["finite_count"],
            "in_front_count": filter_res["in_front_count"],
            "in_image_count": filter_res["in_image_count"],
        },
        "matrix_composition_consistency": {
            "passed": bool(comp_ok and two_stage_ok),
            "max_matrix_err": max_comp_err,
            "two_stage_uv_err_px": two_stage_err,
        },
        "all_passed": bool(known_ok and filter_ok and comp_ok and two_stage_ok),
    }


def audit_local_pairing(data_root: Path, splits: Dict[str, List[str]]) -> Dict[str, Any]:
    """Audit locally available files across radar, image_2, calib, label_2, velodyne."""
    layout = UTILS.resolve_v2x_layout(data_root)

    def _stems(directory: Path, exts: Tuple[str, ...]) -> set:
        if not directory.is_dir():
            return set()
        return {p.stem for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts}

    radar_ids = _stems(layout["radar_dir"], (".bin",))
    image_ids = _stems(layout["image_dir"], UTILS.IMAGE_EXTENSIONS)
    calib_ids = _stems(layout["calib_dir"], (".txt",))
    label_ids = _stems(layout["label_dir"], (".txt",))
    velo_ids = _stems(layout["velodyne_dir"], (".bin",))

    all_local = sorted(radar_ids | image_ids | calib_ids | label_ids)
    complete_4 = sorted(radar_ids & image_ids & calib_ids & label_ids)
    incomplete_local: Dict[str, List[str]] = {}
    for fid in all_local:
        missing = []
        if fid not in radar_ids:
            missing.append("radar")
        if fid not in image_ids:
            missing.append("image_2")
        if fid not in calib_ids:
            missing.append("calib")
        if fid not in label_ids:
            missing.append("label_2")
        if missing:
            incomplete_local[fid] = missing

    empty_label_frames: List[str] = []
    for fid in sorted(label_ids):
        lp = layout["label_dir"] / f"{fid}.txt"
        if not lp.read_text(encoding="utf-8").strip():
            empty_label_frames.append(fid)

    split_coverage: Dict[str, Any] = {}
    complete_set = set(complete_4)
    for split_name, ids in splits.items():
        id_set = set(ids)
        present = len(id_set & complete_set)
        split_coverage[split_name] = {
            "official_count": len(ids),
            "local_complete_4_count": present,
            "missing_from_local_count": len(id_set) - present,
        }

    return {
        "local_counts": {
            "radar": len(radar_ids),
            "image_2": len(image_ids),
            "calib": len(calib_ids),
            "label_2": len(label_ids),
            "velodyne_optional": len(velo_ids),
            "complete_4_modalities": len(complete_4),
            "empty_label_files": len(empty_label_frames),
        },
        "complete_local_frame_ids_sample": complete_4[:20],
        "empty_label_frame_ids_sample": empty_label_frames[:20],
        "incomplete_local_frames": incomplete_local,
        "split_coverage": split_coverage,
    }


def render_radar_on_image(
    image_path: Path,
    uv: np.ndarray,
    depth: np.ndarray,
    calib: Dict[str, np.ndarray],
    annos: Dict[str, Any] | None,
    frame_id: str,
    out_path: Path,
    raw_count: int,
    in_front_count: int,
) -> List[Dict[str, Any]]:
    """Render depth-colored radar points and projected 3D GT boxes onto the image."""
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    H, W = img_bgr.shape[:2]
    canvas = img_bgr.copy()

    # Draw radar points colored by depth (turbo colormap, 0..75m)
    if len(uv) > 0:
        depth_norm = np.clip(depth / 75.0, 0.0, 1.0)
        gray = (depth_norm * 255.0).astype(np.uint8).reshape(-1, 1)
        colors = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO).reshape(-1, 3)
        # Draw farther points first so closer points stay visible
        order = np.argsort(-depth)
        for idx in order:
            u_i = int(round(uv[idx, 0]))
            v_i = int(round(uv[idx, 1]))
            if 0 <= u_i < W and 0 <= v_i < H:
                c = tuple(int(v) for v in colors[idx])
                cv2.circle(canvas, (u_i, v_i), 3, c, -1, lineType=cv2.LINE_AA)

    # Draw GT 3D boxes projected into camera image & measure center residuals vs 2D bbox
    gt_residuals: List[Dict[str, Any]] = []
    if annos is not None and len(annos["name"]) > 0:
        P2_rect = calib["P2"] @ calib["R0_rect"]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i in range(len(annos["name"])):
            raw_cls = str(annos["raw_name"][i])
            loc = annos["location"][i]
            dims_lhw = annos["dimensions"][i]
            rot_y = float(annos["rotation_y"][i])
            b2d = annos["bbox"][i]

            corners_3d = UTILS.camera_box_corners_3d(loc, dims_lhw, rot_y)
            corners_hom = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)
            corners_proj = (P2_rect @ corners_hom.T).T
            if np.all(corners_proj[:, 2] > 0.1):
                pts_2d = (corners_proj[:, :2] / corners_proj[:, 2:3]).astype(np.int32)
                for a, b in edges:
                    cv2.line(canvas, tuple(pts_2d[a]), tuple(pts_2d[b]), (0, 255, 120), 2, cv2.LINE_AA)
                label_pt = (
                    max(4, min(W - 140, int(pts_2d[:, 0].min()))),
                    max(22, min(H - 8, int(pts_2d[:, 1].min()) - 6)),
                )
                cv2.putText(
                    canvas,
                    f"{raw_cls} {loc[2]:.1f}m",
                    label_pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 120),
                    2,
                    cv2.LINE_AA,
                )

            # 3D center (x, y - h/2, z) vs 2D bbox center
            center_3d = np.array([loc[0], loc[1] - dims_lhw[1] / 2.0, loc[2], 1.0])
            c_proj = P2_rect @ center_3d
            if c_proj[2] > 0.1:
                cu, cv_ = float(c_proj[0] / c_proj[2]), float(c_proj[1] / c_proj[2])
                bu, bv = float((b2d[0] + b2d[2]) / 2.0), float((b2d[1] + b2d[3]) / 2.0)
                gt_residuals.append(
                    {
                        "raw_class": raw_cls,
                        "depth_m": round(float(loc[2]), 3),
                        "proj_center_uv": [round(cu, 2), round(cv_, 2)],
                        "bbox2d_center_uv": [round(bu, 2), round(bv, 2)],
                        "residual_px": [round(cu - bu, 2), round(cv_ - bv, 2)],
                    }
                )

    # Top banner
    cv2.rectangle(canvas, (0, 0), (W, 44), (20, 20, 20), -1)
    banner = (
        f"V2X-Radar-V frame {frame_id} | img {W}x{H} | "
        f"radar raw={raw_count} in_front={in_front_count} in_img={len(uv)} | "
        f"GT boxes={len(annos['name']) if annos is not None else 0}"
    )
    cv2.putText(canvas, banner, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f"Failed to write {out_path}")
    return gt_residuals


def array_stats(arr: np.ndarray, names: Tuple[str, ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if arr.shape[0] == 0:
        return {name: None for name in names}
    for i, name in enumerate(names):
        col = arr[:, i]
        out[name] = {
            "min": round(float(np.min(col)), 4),
            "max": round(float(np.max(col)), 4),
            "mean": round(float(np.mean(col)), 4),
            "std": round(float(np.std(col)), 4),
        }
    return out


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = UTILS.resolve_v2x_layout(data_root)
    splits = UTILS.load_official_splits(data_root)
    split_audit = UTILS.audit_splits(splits)
    temporal_audit = UTILS.audit_temporal_metadata(data_root, splits)
    pairing_audit = audit_local_pairing(data_root, splits)

    fid = UTILS.normalize_frame_id(args.frame_id)
    files = UTILS.resolve_frame_files(data_root, fid)
    missing_for_frame = [
        k for k in ("radar", "image", "calib", "label") if not files[k].is_file()
    ]
    if missing_for_frame:
        raise FileNotFoundError(
            f"Frame {fid} is missing required files: {missing_for_frame} "
            f"(resolved under {layout['dataset_root']})"
        )

    img_h, img_w = UTILS.read_image_shape(files["image"])
    calib = UTILS.parse_calib_file(files["calib"])
    radar_raw = UTILS.read_radar_bin(files["radar"], num_features=5)
    radar_r4det = UTILS.format_radar_points_for_r4det(radar_raw, calib["Tr_radar_to_velo"])
    annos = UTILS.parse_label_file(
        files["label"], class_mapping=UTILS.TJ4D_COMPAT_CLASS_MAPPING, allow_missing=False
    )

    proj = UTILS.project_points_to_image(
        radar_raw[:, :3], calib["radar2img"], (img_h, img_w), min_depth=args.min_depth
    )
    self_checks = run_geometry_self_checks(calib, (img_h, img_w))

    proj_img_path = out_dir / "radar_on_image.png"
    gt_residuals = render_radar_on_image(
        image_path=files["image"],
        uv=proj["uv"],
        depth=proj["depth"],
        calib=calib,
        annos=annos,
        frame_id=fid,
        out_path=proj_img_path,
        raw_count=proj["raw_count"],
        in_front_count=proj["in_front_count"],
    )

    # Which official split(s) contain this frame?
    frame_splits = [name for name, ids in splits.items() if fid in set(ids)]

    raw_class_counts: Dict[str, int] = {}
    mapped_class_counts: Dict[str, int] = {}
    for rn, mn in zip(annos["raw_name"].tolist(), annos["name"].tolist()):
        raw_class_counts[rn] = raw_class_counts.get(rn, 0) + 1
        mapped_class_counts[mn] = mapped_class_counts.get(mn, 0) + 1

    report: Dict[str, Any] = {
        "status": "ok" if self_checks["all_passed"] else "geometry_check_failed",
        "data_root": str(data_root),
        "resolved_dataset_root": str(layout["dataset_root"]),
        "frame_id": fid,
        "official_split_membership": frame_splits,
        "files": {
            k: str(files[k]) for k in ("radar", "image", "calib", "label", "velodyne")
            if files[k] is not None and files[k].is_file()
        },
        "image_shape_hw": [img_h, img_w],
        "radar_schema": {
            "raw_columns": list(UTILS.V2X_RADAR_RAW_FIELDS),
            "r4det_columns": list(UTILS.R4DET_RADAR_FIELDS),
            "raw_stats": array_stats(radar_raw, UTILS.V2X_RADAR_RAW_FIELDS),
            "r4det_velo_stats": array_stats(radar_r4det, UTILS.R4DET_RADAR_FIELDS),
        },
        "projection": {
            "matrix_used": "P2 @ R0_rect @ Tr_velo_to_cam @ Tr_radar_to_velo",
            "raw_points": proj["raw_count"],
            "finite_points": proj["finite_count"],
            "in_front_points": proj["in_front_count"],
            "in_image_points": proj["in_image_count"],
            "in_image_ratio": round(proj["in_image_count"] / max(proj["raw_count"], 1), 4),
            "projected_depth_range_m": [
                round(float(proj["depth"].min()), 3),
                round(float(proj["depth"].max()), 3),
            ]
            if len(proj["depth"]) > 0
            else None,
            "output_image": str(proj_img_path),
        },
        "calibration_matrices": {
            key: np.round(calib[key], 6).tolist()
            for key in ("P2", "R0_rect", "Tr_velo_to_cam", "Tr_radar_to_velo")
        },
        "annotations": {
            "num_objects": int(len(annos["name"])),
            "raw_class_counts": raw_class_counts,
            "tj4d_compat_mapped_counts": mapped_class_counts,
            "gt_3d_to_2d_center_residuals": gt_residuals,
        },
        "geometry_self_checks": self_checks,
        "split_audit": split_audit,
        "pairing_audit": pairing_audit,
        "temporal_audit": temporal_audit,
    }

    report_path = out_dir / "inspection.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"INSPECTION_OK frame={fid} status={report['status']}")
    print(f"  image={img_w}x{img_h}, radar raw={proj['raw_count']} -> finite={proj['finite_count']} -> in_front={proj['in_front_count']} -> in_img={proj['in_image_count']}")
    print(f"  gt_objects={len(annos['name'])}, raw_classes={raw_class_counts}")
    print(f"  geometry_self_checks_all_passed={self_checks['all_passed']}")
    print(f"  wrote: {report_path}")
    print(f"  wrote: {proj_img_path}")


if __name__ == "__main__":
    main()
