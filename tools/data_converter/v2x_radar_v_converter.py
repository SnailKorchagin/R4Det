#!/usr/bin/env python3
"""Convert V2X-Radar-V official splits (or selected frames) into MMDet3D .pkl infos.

Shares all parsing, calibration, and class-mapping logic with
``mmdet3d/datasets/v2x_radar_v_utils.py`` and runs on CPU without CUDA.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_utils_module():
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
        description="Generate MMDet3D/R4Det info .pkl files for V2X-Radar-V."
    )
    parser.add_argument(
        "--data-root",
        default="data/V2X-Radar-V",
        help="Path to V2X-Radar-V dataset root (containing training/ and ImageSets/).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for .pkl files (defaults to resolved dataset root).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Official ImageSets split names to convert.",
    )
    parser.add_argument(
        "--frame-ids",
        nargs="+",
        default=None,
        help="Optional explicit frame IDs; writes a single custom pkl named via --custom-name.",
    )
    parser.add_argument(
        "--custom-name",
        default="v2x_radar_v_infos_custom.pkl",
        help="Output filename when --frame-ids is specified.",
    )
    parser.add_argument(
        "--class-scheme",
        choices=["tj4d_compat", "v2x_benchmark", "raw"],
        default="tj4d_compat",
        help="Class mapping scheme stored in annos['name'] (raw_name is always preserved).",
    )
    parser.add_argument(
        "--only-available",
        action="store_true",
        help="Skip split frames not present locally instead of raising FileNotFoundError.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Build unlabeled infos (annos=None) for sensor-only inference checks.",
    )
    parser.add_argument(
        "--allow-missing-label",
        action="store_true",
        help="Allow missing label_2 files (sets annos=None for those frames).",
    )
    return parser.parse_args()


def get_class_mapping(scheme: str) -> Optional[Dict[str, Optional[str]]]:
    if scheme == "tj4d_compat":
        return UTILS.TJ4D_COMPAT_CLASS_MAPPING
    if scheme == "v2x_benchmark":
        return UTILS.V2X_BENCHMARK_CLASS_MAPPING
    return None


def convert_frame_list(
    data_root: Path,
    frame_ids: List[str],
    class_mapping: Optional[Dict[str, Optional[str]]],
    with_label: bool,
    allow_missing_label: bool,
    only_available: bool,
) -> Dict[str, Any]:
    infos: List[Dict[str, Any]] = []
    skipped_missing: List[str] = []
    empty_gt_frames: List[str] = []
    unlabeled_frames: List[str] = []
    raw_class_counts: Dict[str, int] = {}
    mapped_class_counts: Dict[str, int] = {}

    for fid in frame_ids:
        files = UTILS.resolve_frame_files(data_root, fid)
        required = ["radar", "image", "calib"]
        if with_label and not allow_missing_label:
            required.append("label")
        missing = [k for k in required if not files[k].is_file()]
        if missing:
            if only_available:
                skipped_missing.append(fid)
                continue
            raise FileNotFoundError(
                f"Frame {fid} missing required files {missing} under {data_root}"
            )

        info = UTILS.build_frame_info(
            data_root=data_root,
            frame_id=fid,
            with_label=with_label,
            allow_missing_label=allow_missing_label,
            class_mapping=class_mapping,
        )
        infos.append(info)

        annos = info.get("annos")
        if annos is None:
            unlabeled_frames.append(fid)
        elif len(annos["name"]) == 0:
            empty_gt_frames.append(fid)
        else:
            for rn, mn in zip(annos["raw_name"].tolist(), annos["name"].tolist()):
                raw_class_counts[rn] = raw_class_counts.get(rn, 0) + 1
                mapped_class_counts[mn] = mapped_class_counts.get(mn, 0) + 1

    return {
        "infos": infos,
        "summary": {
            "requested": len(frame_ids),
            "converted": len(infos),
            "skipped_missing_local": len(skipped_missing),
            "empty_gt_frames": empty_gt_frames,
            "unlabeled_frames": unlabeled_frames,
            "raw_class_counts": raw_class_counts,
            "mapped_class_counts": mapped_class_counts,
        },
    }


def main() -> None:
    args = parse_args()
    layout = UTILS.resolve_v2x_layout(args.data_root)
    dataset_root = layout["dataset_root"]
    out_dir = Path(args.out_dir) if args.out_dir else dataset_root
    out_dir.mkdir(parents=True, exist_ok=True)
    class_mapping = get_class_mapping(args.class_scheme)

    report: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "class_scheme": args.class_scheme,
        "outputs": {},
    }

    if args.frame_ids is not None:
        fids = [UTILS.normalize_frame_id(x) for x in args.frame_ids]
        result = convert_frame_list(
            data_root=dataset_root,
            frame_ids=fids,
            class_mapping=class_mapping,
            with_label=not args.no_labels,
            allow_missing_label=args.allow_missing_label,
            only_available=args.only_available,
        )
        out_path = out_dir / args.custom_name
        with open(out_path, "wb") as f:
            pickle.dump(result["infos"], f)
        report["outputs"][str(out_path)] = result["summary"]
        print(f"WROTE {out_path}: {json.dumps(result['summary'])}")
        return

    splits = UTILS.load_official_splits(dataset_root)
    for split_name in args.splits:
        if split_name not in splits:
            raise FileNotFoundError(
                f"Split '{split_name}' not found under {layout['imagesets_dir']}"
            )
        result = convert_frame_list(
            data_root=dataset_root,
            frame_ids=splits[split_name],
            class_mapping=class_mapping,
            with_label=not args.no_labels,
            allow_missing_label=args.allow_missing_label,
            only_available=args.only_available,
        )
        out_path = out_dir / f"v2x_radar_v_infos_{split_name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(result["infos"], f)
        report["outputs"][str(out_path)] = result["summary"]
        print(f"WROTE {out_path}: {json.dumps(result['summary'])}")


if __name__ == "__main__":
    main()
