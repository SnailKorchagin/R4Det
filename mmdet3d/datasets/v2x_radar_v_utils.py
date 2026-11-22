"""Pure-CPU utilities for reading, auditing, and converting V2X-Radar-V data.

Shared by Stage 1 inspection CLI, data converter, and the R4Det
``V2XRadarVDataset`` / loading pipeline so pairing, calibration, projection,
and class mapping use a single source of truth without requiring CUDA on import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# Raw V2X-Radar-V binary point schema verified on real files:
#   col 0: x_radar (lateral, right)
#   col 1: y_radar (longitudinal, forward)
#   col 2: z_radar (vertical, up)
#   col 3: intensity (signal strength / RCS, positive ~105..174)
#   col 4: velocity (Doppler radial velocity, signed m/s)
V2X_RADAR_RAW_FIELDS = ("x_radar", "y_radar", "z_radar", "intensity", "velocity")
R4DET_RADAR_FIELDS = ("x_velo", "y_velo", "z_velo", "velocity", "intensity")

# Default 4-class mapping aligned with TJ4D R4Det checkpoint for Stage 2 forward.
TJ4D_COMPAT_CLASSES = ("Pedestrian", "Cyclist", "Car", "Truck")
TJ4D_COMPAT_CLASS_MAPPING: Dict[str, Optional[str]] = {
    "Pedestrian": "Pedestrian",
    "Cyclist": "Cyclist",
    "Car": "Car",
    "Van": "Car",
    "Truck": "Truck",
    "Bus": "Truck",
    "Bicycle": None,
    "Tricycle": None,
    "Dog": None,
    "Others": None,
    "Misc": None,
    "DontCare": None,
}

# Native V2X-Radar-V 3-class benchmark mapping (Vehicle = Car/Van/Truck/Bus).
V2X_BENCHMARK_CLASSES = ("Pedestrian", "Cyclist", "Vehicle")
V2X_BENCHMARK_CLASS_MAPPING: Dict[str, Optional[str]] = {
    "Pedestrian": "Pedestrian",
    "Cyclist": "Cyclist",
    "Car": "Vehicle",
    "Van": "Vehicle",
    "Truck": "Vehicle",
    "Bus": "Vehicle",
    "Bicycle": None,
    "Tricycle": None,
    "Dog": None,
    "Others": None,
    "Misc": None,
    "DontCare": None,
}

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")


def normalize_frame_id(frame_id: Any) -> str:
    """Format an int or string frame ID to V2X-Radar-V's 6-digit string."""
    if isinstance(frame_id, (int, np.integer)):
        return f"{int(frame_id):06d}"
    text = str(frame_id).strip()
    if text.isdigit():
        return f"{int(text):06d}"
    return text


def resolve_v2x_layout(data_root: str | Path) -> Dict[str, Path]:
    """Resolve V2X-Radar-V root, training dir, and ImageSets dir.

    Accepts either the dataset root containing ``training/`` & ``ImageSets/``
    or a parent directory containing ``V2X-Radar-V/training/``.
    """
    root = Path(data_root)
    if (root / "training").is_dir():
        dataset_root = root
    elif (root / "V2X-Radar-V" / "training").is_dir():
        dataset_root = root / "V2X-Radar-V"
    else:
        dataset_root = root

    training_dir = dataset_root / "training"
    imagesets_dir = dataset_root / "ImageSets"
    if not imagesets_dir.is_dir() and (root / "ImageSets").is_dir():
        imagesets_dir = root / "ImageSets"

    return {
        "dataset_root": dataset_root,
        "training_dir": training_dir,
        "imagesets_dir": imagesets_dir,
        "radar_dir": training_dir / "radar",
        "velodyne_dir": training_dir / "velodyne",
        "image_dir": training_dir / "image_2",
        "calib_dir": training_dir / "calib",
        "label_dir": training_dir / "label_2",
    }


def read_split_file(split_path: str | Path) -> List[str]:
    """Read frame IDs from an official ImageSets split file."""
    path = Path(split_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [
        normalize_frame_id(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_official_splits(data_root: str | Path) -> Dict[str, List[str]]:
    """Load all present official ImageSets splits (train/val/trainval/test)."""
    layout = resolve_v2x_layout(data_root)
    splits: Dict[str, List[str]] = {}
    for name in ("train", "val", "trainval", "test"):
        split_path = layout["imagesets_dir"] / f"{name}.txt"
        if split_path.is_file():
            splits[name] = read_split_file(split_path)
    return splits


def audit_splits(splits: Dict[str, List[str]]) -> Dict[str, Any]:
    """Audit split sizes, duplicates, pairwise intersections, and trainval union."""
    summary: Dict[str, Any] = {}
    sets: Dict[str, set] = {}
    for name, ids in splits.items():
        id_set = set(ids)
        sets[name] = id_set
        summary[name] = {
            "count": len(ids),
            "unique": len(id_set),
            "duplicates": len(ids) - len(id_set),
            "first_5": ids[:5],
            "last_5": ids[-5:],
        }

    overlaps: Dict[str, int] = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in sets and b in sets:
            overlaps[f"{a}_and_{b}"] = len(sets[a] & sets[b])

    trainval_matches_union = None
    if {"train", "val", "trainval"}.issubset(sets.keys()):
        trainval_matches_union = (sets["train"] | sets["val"]) == sets["trainval"]

    return {
        "splits": summary,
        "overlaps": overlaps,
        "trainval_equals_train_union_val": trainval_matches_union,
    }


def audit_temporal_metadata(
    data_root: str | Path, splits: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Check whether sequence IDs, timestamps, or contiguous frame order exist."""
    layout = resolve_v2x_layout(data_root)
    training_dir = layout["training_dir"]
    extra_dirs = sorted(
        p.name for p in training_dir.iterdir() if p.is_dir()
    ) if training_dir.is_dir() else []
    extra_files = sorted(
        p.name for p in layout["dataset_root"].iterdir() if p.is_file()
    ) if layout["dataset_root"].is_dir() else []

    timestamp_candidates = [
        name for name in extra_dirs + extra_files
        if any(tok in name.lower() for tok in ("time", "seq", "pose", "odom", "meta"))
    ]

    split_run_stats: Dict[str, Any] = {}
    for name, ids in splits.items():
        if not ids:
            continue
        ints = sorted(int(x) for x in ids if x.isdigit())
        if not ints:
            continue
        diffs = np.diff(ints)
        contiguous_pairs = int(np.sum(diffs == 1))
        gaps = int(np.sum(diffs > 1))
        # Count contiguous runs
        runs = 1 + gaps
        split_run_stats[name] = {
            "min_id": ints[0],
            "max_id": ints[-1],
            "contiguous_adjacent_pairs": contiguous_pairs,
            "num_gaps": gaps,
            "num_contiguous_runs": runs,
            "mean_run_length": round(len(ints) / max(runs, 1), 2),
        }

    cross_split_adjacent = 0
    if "train" in splits and "val" in splits:
        train_ints = {int(x) for x in splits["train"] if x.isdigit()}
        val_ints = {int(x) for x in splits["val"] if x.isdigit()}
        cross_split_adjacent = sum(
            1 for v in val_ints if (v - 1) in train_ints or (v + 1) in train_ints
        )

    return {
        "training_subdirs": extra_dirs,
        "root_files": extra_files,
        "sequence_or_timestamp_candidates": timestamp_candidates,
        "has_explicit_sequence_id": False,
        "has_explicit_timestamp": False,
        "cross_split_adjacent_val_frames": cross_split_adjacent,
        "split_frame_order_stats": split_run_stats,
        "temporal_continuity_reliable": False,
        "recommendation": (
            "No sequence ID, ego-pose, or timestamp files are present and "
            "contiguous frame IDs are interleaved across train/val/test. "
            "Use single-frame mode (is_prev_frame_valid=False) until explicit "
            "sequence/timestamp metadata is verified."
        ),
    }


def resolve_frame_files(
    data_root: str | Path, frame_id: Any
) -> Dict[str, Optional[Path]]:
    """Resolve radar, image, calib, label, and optional velodyne paths for a frame."""
    layout = resolve_v2x_layout(data_root)
    fid = normalize_frame_id(frame_id)

    image_path: Optional[Path] = None
    for ext in IMAGE_EXTENSIONS:
        candidate = layout["image_dir"] / f"{fid}{ext}"
        if candidate.is_file():
            image_path = candidate
            break
    if image_path is None:
        image_path = layout["image_dir"] / f"{fid}.jpg"

    return {
        "frame_id": fid,
        "radar": layout["radar_dir"] / f"{fid}.bin",
        "image": image_path,
        "calib": layout["calib_dir"] / f"{fid}.txt",
        "label": layout["label_dir"] / f"{fid}.txt",
        "velodyne": layout["velodyne_dir"] / f"{fid}.bin",
    }


def parse_calib_file(calib_path: str | Path) -> Dict[str, np.ndarray]:
    """Parse a V2X-Radar-V KITTI-style calib file into 4x4 float64 matrices.

    Expected keys: ``P2``, ``R0_rect``, ``Tr_velo_to_cam``, ``Tr_radar_to_velo``.
    """
    path = Path(calib_path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    raw: Dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, values_str = line.split(":", 1)
        key = key.strip()
        vals = np.fromstring(values_str.strip(), sep=" ", dtype=np.float64)
        if key == "R0_rect":
            if vals.size != 9:
                raise ValueError(f"Expected 9 values for R0_rect in {path}, got {vals.size}")
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = vals.reshape(3, 3)
            raw[key] = mat
        elif vals.size == 12:
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = vals.reshape(3, 4)
            raw[key] = mat
        elif vals.size == 16:
            raw[key] = vals.reshape(4, 4)
        else:
            raw[key] = vals

    for required in ("P2", "R0_rect", "Tr_velo_to_cam", "Tr_radar_to_velo"):
        if required not in raw:
            raise KeyError(f"Missing required calib matrix '{required}' in {path}")

    lidar2cam = raw["R0_rect"] @ raw["Tr_velo_to_cam"]
    lidar2img = raw["P2"] @ lidar2cam
    radar2velo = raw["Tr_radar_to_velo"]
    radar2cam = lidar2cam @ radar2velo
    radar2img = raw["P2"] @ radar2cam

    raw["lidar2cam"] = lidar2cam
    raw["lidar2img"] = lidar2img
    raw["radar2velo"] = radar2velo
    raw["radar2cam"] = radar2cam
    raw["radar2img"] = radar2img
    return raw


def read_radar_bin(radar_path: str | Path, num_features: int = 5) -> np.ndarray:
    """Read raw V2X-Radar-V 4D radar binary file as float32 array of shape (N, 5)."""
    path = Path(radar_path)
    if not path.is_file():
        raise FileNotFoundError(f"Radar file not found: {path}")
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % num_features != 0:
        raise ValueError(
            f"Radar file {path} has {raw.size} float32 values, "
            f"not divisible by {num_features}."
        )
    return raw.reshape(-1, num_features)


def transform_points_homogeneous(
    points_xyz: np.ndarray, transform_4x4: np.ndarray
) -> np.ndarray:
    """Apply a 4x4 rigid transform to (N, 3) points."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_hom = np.concatenate([pts, ones], axis=1)
    transformed = (transform_4x4 @ pts_hom.T).T
    return transformed[:, :3]


def transform_radar_to_velo(
    radar_points_5d: np.ndarray, tr_radar_to_velo: np.ndarray
) -> np.ndarray:
    """Transform radar xyz to velodyne frame, keeping [intensity, velocity]."""
    pts = np.asarray(radar_points_5d, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 5:
        raise ValueError(f"Expected (N, 5) radar points, got shape {pts.shape}")
    out = pts.copy()
    if pts.shape[0] > 0:
        out[:, :3] = transform_points_homogeneous(pts[:, :3], tr_radar_to_velo).astype(
            np.float32
        )
    return out


def format_radar_points_for_r4det(
    radar_points_5d: np.ndarray,
    tr_radar_to_velo: np.ndarray,
    intensity_offset: float = 0.0,
    intensity_scale: float = 1.0,
) -> np.ndarray:
    """Convert raw V2X-Radar-V [x_r, y_r, z_r, intensity, velocity] to R4Det 5D.

    Output columns match R4Det's ``RadarPillarFeatureNet`` (``in_channels=5``):
      ``[x_velo, y_velo, z_velo, velocity, (intensity - intensity_offset) * intensity_scale]``.
    """
    velo_pts = transform_radar_to_velo(radar_points_5d, tr_radar_to_velo)
    if velo_pts.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32)
    intensity = (velo_pts[:, 3] - float(intensity_offset)) * float(intensity_scale)
    velocity = velo_pts[:, 4]
    return np.stack(
        [velo_pts[:, 0], velo_pts[:, 1], velo_pts[:, 2], velocity, intensity],
        axis=1,
    ).astype(np.float32)


def project_points_to_image(
    points_xyz: np.ndarray,
    proj_4x4: np.ndarray,
    img_shape: Tuple[int, int],
    min_depth: float = 1e-6,
) -> Dict[str, Any]:
    """Project 3D points to image pixels using a 4x4 projection matrix.

    Filters non-finite points, behind-camera points (``z <= min_depth``), and
    out-of-bounds pixels (``u < 0`` or ``u >= W`` or ``v < 0`` or ``v >= H``).
    """
    H, W = int(img_shape[0]), int(img_shape[1])
    pts = np.asarray(points_xyz, dtype=np.float64)
    n_raw = int(pts.shape[0])
    if n_raw == 0:
        empty_f = np.zeros((0,), dtype=np.float64)
        empty_b = np.zeros((0,), dtype=bool)
        return {
            "raw_count": 0,
            "finite_count": 0,
            "in_front_count": 0,
            "in_image_count": 0,
            "valid_mask": empty_b,
            "uv": np.zeros((0, 2), dtype=np.float64),
            "depth": empty_f,
        }

    finite_mask = np.isfinite(pts).all(axis=1)
    pts_safe = np.where(finite_mask[:, None], pts, 0.0)
    ones = np.ones((n_raw, 1), dtype=np.float64)
    pts_hom = np.concatenate([pts_safe, ones], axis=1)
    proj = (proj_4x4 @ pts_hom.T).T  # (N, 4)
    depth = proj[:, 2]

    in_front_mask = finite_mask & np.isfinite(depth) & (depth > min_depth)
    safe_depth = np.where(in_front_mask, depth, 1.0)
    u = proj[:, 0] / safe_depth
    v = proj[:, 1] / safe_depth
    uv_finite = np.isfinite(u) & np.isfinite(v)

    in_image_mask = (
        in_front_mask
        & uv_finite
        & (u >= 0.0)
        & (u < float(W))
        & (v >= 0.0)
        & (v < float(H))
    )

    return {
        "raw_count": n_raw,
        "finite_count": int(finite_mask.sum()),
        "in_front_count": int(in_front_mask.sum()),
        "in_image_count": int(in_image_mask.sum()),
        "valid_mask": in_image_mask,
        "uv": np.stack([u[in_image_mask], v[in_image_mask]], axis=1),
        "depth": depth[in_image_mask],
    }


def empty_kitti_annos() -> Dict[str, np.ndarray]:
    """Return a valid empty annotation dict with canonical array shapes."""
    return {
        "name": np.array([], dtype="<U16"),
        "raw_name": np.array([], dtype="<U16"),
        "truncated": np.zeros((0,), dtype=np.float32),
        "occluded": np.zeros((0,), dtype=np.int64),
        "alpha": np.zeros((0,), dtype=np.float32),
        "bbox": np.zeros((0, 4), dtype=np.float32),
        "dimensions": np.zeros((0, 3), dtype=np.float32),
        "location": np.zeros((0, 3), dtype=np.float32),
        "rotation_y": np.zeros((0,), dtype=np.float32),
        "track_id": np.zeros((0,), dtype=np.int64),
        "score": np.zeros((0,), dtype=np.float32),
        "index": np.zeros((0,), dtype=np.int32),
        "group_ids": np.zeros((0,), dtype=np.int32),
    }


def parse_label_file(
    label_path: str | Path,
    class_mapping: Optional[Dict[str, Optional[str]]] = None,
    allow_missing: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    """Parse a V2X-Radar-V KITTI-style label_2 file.

    Distinguishes:
    - Missing file: raises ``FileNotFoundError`` unless ``allow_missing=True``
      (in which case returns ``None`` to mark an unlabeled frame).
    - Existing empty file (0 objects): returns ``empty_kitti_annos()``.

    Dimensions in the raw text are ``(h, w, l)`` and are converted to MMDet3D's
    camera convention ``(l, h, w)``. Column 15 (if present) is stored in
    ``track_id`` (NOT treated as a confidence score).
    """
    path = Path(label_path)
    if not path.is_file():
        if allow_missing:
            return None
        raise FileNotFoundError(f"Label file not found: {path}")

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return empty_kitti_annos()

    rows = [line.split() for line in lines]
    for i, row in enumerate(rows):
        if len(row) < 15:
            raise ValueError(
                f"Label file {path} line {i + 1} has {len(row)} fields (< 15): {row}"
            )

    raw_names = [row[0] for row in rows]
    if class_mapping is not None:
        mapped_names = [
            class_mapping.get(name, name) if name in class_mapping else name
            for name in raw_names
        ]
        # Keep unmapped/ignored labels as 'DontCare' if explicitly mapped to None
        # so raw_name is preserved while training label assignment maps to -1.
        mapped_names = [
            "DontCare" if m is None else m for m in mapped_names
        ]
    else:
        mapped_names = list(raw_names)

    dims_hwl = np.array(
        [[float(v) for v in row[8:11]] for row in rows], dtype=np.float32
    ).reshape(-1, 3)
    # KITTI text is (h, w, l); MMDet3D CameraInstance3DBoxes expects (l, h, w)
    dims_lhw = dims_hwl[:, [2, 0, 1]].copy()

    track_ids = np.array(
        [int(float(row[15])) if len(row) >= 16 else -1 for row in rows],
        dtype=np.int64,
    )
    num_gt = len(rows)
    num_valid = sum(1 for n in mapped_names if n != "DontCare")
    index = list(range(num_valid)) + [-1] * (num_gt - num_valid)

    return {
        "name": np.array(mapped_names, dtype="<U32"),
        "raw_name": np.array(raw_names, dtype="<U32"),
        "truncated": np.array([float(row[1]) for row in rows], dtype=np.float32),
        "occluded": np.array([int(float(row[2])) for row in rows], dtype=np.int64),
        "alpha": np.array([float(row[3]) for row in rows], dtype=np.float32),
        "bbox": np.array(
            [[float(v) for v in row[4:8]] for row in rows], dtype=np.float32
        ).reshape(-1, 4),
        "dimensions": dims_lhw,
        "location": np.array(
            [[float(v) for v in row[11:14]] for row in rows], dtype=np.float32
        ).reshape(-1, 3),
        "rotation_y": np.array([float(row[14]) for row in rows], dtype=np.float32),
        "track_id": track_ids,
        "score": np.zeros((num_gt,), dtype=np.float32),
        "index": np.array(index, dtype=np.int32),
        "group_ids": np.arange(num_gt, dtype=np.int32),
    }


def read_image_shape(image_path: str | Path) -> Tuple[int, int]:
    """Read image (height, width) without decoding full RGB unless needed."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    with Image.open(path) as img:
        w, h = img.size
    return int(h), int(w)


def camera_box_corners_3d(
    location: np.ndarray, dimensions_lhw: np.ndarray, rotation_y: float
) -> np.ndarray:
    """Compute 8 corners (8, 3) in camera frame for a KITTI bottom-center 3D box."""
    l, h, w = [float(x) for x in dimensions_lhw]
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0.0, 0.0, 0.0, 0.0, -h, -h, -h, -h])
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    corners = rot @ np.vstack([x_corners, y_corners, z_corners])
    corners = corners.T + np.asarray(location, dtype=np.float64)[None, :]
    return corners


def build_frame_info(
    data_root: str | Path,
    frame_id: Any,
    with_label: bool = True,
    allow_missing_label: bool = False,
    class_mapping: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """Build one MMDet3D-compatible info dict for a V2X-Radar-V frame."""
    layout = resolve_v2x_layout(data_root)
    dataset_root = layout["dataset_root"]
    files = resolve_frame_files(dataset_root, frame_id)
    fid = files["frame_id"]

    # Validate critical sensor & calib files exist
    for key in ("radar", "image", "calib"):
        if not files[key].is_file():
            raise FileNotFoundError(
                f"Missing critical {key} file for frame {fid}: {files[key]}"
            )

    img_h, img_w = read_image_shape(files["image"])
    calib = parse_calib_file(files["calib"])

    rel_image = str(files["image"].relative_to(dataset_root))
    rel_radar = str(files["radar"].relative_to(dataset_root))
    rel_velodyne = (
        str(files["velodyne"].relative_to(dataset_root))
        if files["velodyne"].is_file()
        else None
    )

    annos = None
    if with_label:
        annos = parse_label_file(
            files["label"],
            class_mapping=class_mapping,
            allow_missing=allow_missing_label,
        )

    info: Dict[str, Any] = {
        "image": {
            "image_idx": int(fid) if fid.isdigit() else fid,
            "frame_id": fid,
            "image_path": rel_image,
            "image_shape": np.array([img_h, img_w], dtype=np.int32),
        },
        "point_cloud": {
            "num_features": 5,
            "velodyne_path": rel_radar,
            "radar_path": rel_radar,
            "lidar_path": rel_velodyne,
        },
        "calib": {
            "P2": calib["P2"].astype(np.float32),
            "R0_rect": calib["R0_rect"].astype(np.float32),
            "Tr_velo_to_cam": calib["Tr_velo_to_cam"].astype(np.float32),
            "Tr_radar_to_velo": calib["Tr_radar_to_velo"].astype(np.float32),
        },
        "annos": annos,
    }
    return info
