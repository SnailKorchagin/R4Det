#!/usr/bin/env python3
"""Stage 5 explicit V2X-Radar-V evaluation, training-curve, and local showcase CLI.

Never overwrites repository source files. From a single command with
``--config``, ``--checkpoint``, and ``--ann-file``, this script:
1. Runs GPU inference on the specified split and saves ``results.pkl``.
2. Evaluates 3D & BEV AP (R40) via ``mmdet3d/core/evaluation/v2x_radar_v_eval.py``
   under both ``v2x_benchmark`` (Vehicle/Pedestrian/Cyclist) and ``tj4d_compat``
   (Pedestrian/Cyclist/Car/Truck) protocols and saves ``eval_metrics.json``.
3. Plots ``training_curves.png`` if ``--train-metrics`` JSONL is provided.
4. Renders per-frame original camera, radar-on-image projection, and BEV
   prediction-vs-GT views and writes a self-contained local interactive viewer
   at ``<out-dir>/showcase/index.html``.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/r4det-v2x-eval-mpl")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel

from mmdet3d.core.evaluation.v2x_radar_v_eval import evaluate_v2x_radar_v
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.datasets.v2x_radar_v_utils import (
    parse_calib_file,
    project_points_to_image,
    read_radar_bin,
    resolve_frame_files,
)
from mmdet3d.models import build_model
from tools.inspect_v2x_radar_v import render_radar_on_image
from tools.verify_v2x_radar_v_r4det import render_v2x_bev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate R4Det on V2X-Radar-V and generate local showcase + training curves."
    )
    parser.add_argument(
        "--config",
        default="configs/r4det/V2X-Radar-V-R4Det_single_frame.py",
        help="Path to V2X-Radar-V R4Det config.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth file to evaluate.")
    parser.add_argument("--ann-file", default=None, help="Override val/test ann_file .pkl.")
    parser.add_argument("--out-dir", default="work_dirs/v2x_radar_v/eval", help="Output directory.")
    parser.add_argument(
        "--protocol",
        choices=["both", "v2x_benchmark", "tj4d_compat"],
        default="both",
        help="Evaluation protocol(s) to compute.",
    )
    parser.add_argument("--score-thr", type=float, default=0.0, help="Score threshold for AP eval.")
    parser.add_argument("--vis-score-thr", type=float, default=0.05, help="Score threshold for BEV visualization.")
    parser.add_argument("--max-vis-frames", type=int, default=20, help="Max frames to render for HTML showcase.")
    parser.add_argument("--train-metrics", default=None, help="Optional train_metrics.jsonl path for training curves.")
    return parser.parse_args()


def plot_training_curves(metrics_jsonl: Path, out_path: Path) -> Optional[Dict[str, Any]]:
    """Plot total loss, LR, and grad_norm curves from train_metrics.jsonl."""
    if not metrics_jsonl.is_file():
        return None
    rows = [
        json.loads(line)
        for line in metrics_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return None

    iters = [r["iter"] for r in rows]
    losses = [r["total_loss"] for r in rows]
    lrs = [r["lr"] for r in rows]
    grads = [r.get("grad_norm", 0.0) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=130)
    axes[0].plot(iters, losses, marker="o", linewidth=1.8, color="#1f77b4")
    axes[0].set_title("Total Training Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, grads, marker="s", linewidth=1.8, color="#d62728")
    axes[1].set_title("Gradient Norm (pre-clip)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("L2 Norm")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iters, lrs, marker="^", linewidth=1.8, color="#2ca02c")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("LR")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"V2X-Radar-V R4Det Training Curves ({metrics_jsonl.parent.name})", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)
    return {
        "num_points": len(rows),
        "start_loss": losses[0],
        "end_loss": losses[-1],
        "min_loss": min(losses),
        "plot_path": str(out_path),
    }


def build_showcase_html(
    showcase_dir: Path,
    frame_entries: List[Dict[str, Any]],
    eval_report: Dict[str, Any],
    curve_rel_path: Optional[str],
) -> Path:
    """Generate a self-contained local interactive HTML page for frame-by-frame inspection."""
    data_json = json.dumps(frame_entries, ensure_ascii=False)
    metrics_json = html.escape(json.dumps(eval_report["metrics"], indent=2))
    curve_html = (
        f'<section class="card"><h2>Training Curves</h2>'
        f'<img src="{curve_rel_path}" alt="Training Curves" style="width:100%;max-width:1100px;border-radius:8px;"/>'
        f'</section>'
        if curve_rel_path
        else ""
    )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>R4Det x V2X-Radar-V Evaluation Showcase</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 20px; background: #11161d; color: #e6edf3; }}
  h1, h2 {{ margin: 0 0 12px 0; }}
  .meta {{ color: #9aa7b4; font-size: 13px; margin-bottom: 16px; }}
  .controls {{ display: flex; gap: 10px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }}
  button, select {{ background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 14px; }}
  button:hover {{ background: #30363d; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 14px; margin-bottom: 18px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px; margin-bottom: 16px; }}
  .card img {{ width: 100%; border-radius: 6px; display: block; background: #0d1117; }}
  pre {{ background: #0d1117; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #30363d; padding: 6px 8px; text-align: left; }}
</style>
</head>
<body>
  <h1>R4Det × V2X-Radar-V 评测与逐帧展示</h1>
  <div class="meta">
    Config: <code>{html.escape(eval_report['config'])}</code> |
    Checkpoint: <code>{html.escape(eval_report['checkpoint'])}</code> |
    Split: <code>{html.escape(eval_report['ann_file'])}</code> ({eval_report['num_frames']} frames)
  </div>

  <div class="controls">
    <button onclick="step(-1)">← 上一帧</button>
    <select id="frameSelect" onchange="render(parseInt(this.value, 10))"></select>
    <button onclick="step(1)">下一帧 →</button>
    <span id="frameInfo" class="meta" style="margin:0;"></span>
  </div>

  <div class="grid">
    <div class="card"><h2>原图 (Camera)</h2><img id="imgOrig" /></div>
    <div class="card"><h2>4D 雷达投影 + 3D GT</h2><img id="imgProj" /></div>
    <div class="card"><h2>BEV 预测 (绿) vs GT (红)</h2><img id="imgBev" /></div>
  </div>

  <div class="card">
    <h2>当前帧 Top 预测与 GT 摘要</h2>
    <div id="frameTable"></div>
  </div>

  {curve_html}

  <div class="card">
    <h2>评测指标 (AP R40)</h2>
    <pre>{metrics_json}</pre>
  </div>

<script>
const FRAMES = {data_json};
let idx = 0;
const sel = document.getElementById('frameSelect');
FRAMES.forEach((f, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `Frame ${{f.frame_id}} (GT=${{f.gt_count}}, Pred=${{f.pred_count}})`;
  sel.appendChild(opt);
}});
function step(d) {{
  if (!FRAMES.length) return;
  idx = (idx + d + FRAMES.length) % FRAMES.length;
  sel.value = idx;
  render(idx);
}}
function render(i) {{
  idx = i;
  const f = FRAMES[i];
  if (!f) return;
  document.getElementById('imgOrig').src = f.orig_img;
  document.getElementById('imgProj').src = f.proj_img;
  document.getElementById('imgBev').src = f.bev_img;
  document.getElementById('frameInfo').textContent = `Frame ${{f.frame_id}} | GT: ${{f.gt_count}} | Pred: ${{f.pred_count}} | Max Score: ${{f.score_max ?? 'N/A'}}`;
  let rows = '<table><thead><tr><th>Rank</th><th>Class</th><th>Score</th><th>Box [x, y, z, dx, dy, dz, yaw]</th></tr></thead><tbody>';
  f.top5.forEach((p, k) => {{
    rows += `<tr><td>${{k+1}}</td><td>${{p.name}}</td><td>${{p.score.toFixed(4)}}</td><td><code>${{JSON.stringify(p.box)}}</code></td></tr>`;
  }});
  rows += '</tbody></table>';
  document.getElementById('frameTable').innerHTML = rows;
}}
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') step(-1);
  if (e.key === 'ArrowRight') step(1);
}});
if (FRAMES.length) render(0);
</script>
</body>
</html>
"""
    index_path = showcase_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    return index_path


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R4Det evaluation.")

    logging.getLogger("mmcv").setLevel(logging.WARNING)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    showcase_dir = out_dir / "showcase"
    assets_dir = showcase_dir / "frames"
    assets_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.meta_info = {
        "figures_path": str(out_dir / "figures"),
        "project_name": "V2X-Radar-V-R4Det-eval",
    }
    if args.ann_file:
        cfg.data.val.ann_file = args.ann_file
        cfg.data.test.ann_file = args.ann_file

    dataset = build_dataset(cfg.data.val)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    incomp = model.load_state_dict(state_dict, strict=False)
    model.CLASSES = ckpt.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model_dp = MMDataParallel(model.cuda().eval(), device_ids=[0])

    gt_records: List[Dict[str, Any]] = []
    pred_records: List[Dict[str, Any]] = []
    raw_results: List[Dict[str, Any]] = []
    frame_entries: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    data_root = Path(cfg.data_root)

    for idx, batch in enumerate(loader):
        info = dataset.data_infos[idx]
        fid = info["image"].get("frame_id") or f"{int(info['image']['image_idx']):06d}"

        with torch.inference_mode():
            out = model_dp(return_loss=False, rescale=True, **batch)

        pred = out[0]["pts_bbox"] if "pts_bbox" in out[0] else out[0]
        p_boxes = pred["boxes_3d"].tensor.detach().cpu().numpy()
        p_scores = pred["scores_3d"].detach().cpu().numpy()
        p_labels = pred["labels_3d"].detach().cpu().numpy()
        p_names = [
            dataset.CLASSES[int(l)] if 0 <= int(l) < len(dataset.CLASSES) else "Unknown"
            for l in p_labels
        ]

        gt_ann = dataset.get_ann_info(idx)
        g_boxes = gt_ann["gt_bboxes_3d"].tensor.detach().cpu().numpy()
        g_names = [str(n) for n in gt_ann.get("raw_names", gt_ann["gt_names"]).tolist()]

        gt_records.append({"frame_id": fid, "boxes_3d": g_boxes, "names": g_names})
        pred_records.append(
            {
                "frame_id": fid,
                "boxes_3d": p_boxes,
                "scores_3d": p_scores,
                "names": p_names,
            }
        )
        raw_results.append(
            {
                "frame_id": fid,
                "boxes_3d": p_boxes,
                "scores_3d": p_scores,
                "labels_3d": p_labels,
            }
        )

        # Render showcase panels for up to max_vis_frames
        if len(frame_entries) < args.max_vis_frames:
            files = resolve_frame_files(data_root, fid)
            calib = parse_calib_file(files["calib"])
            radar_raw = read_radar_bin(files["radar"], num_features=5)
            img_bgr = cv2.imread(str(files["image"]), cv2.IMREAD_COLOR)
            H, W = img_bgr.shape[:2]

            # 1. Original image copy (resized to 960px wide for lightweight local browsing)
            orig_dst = assets_dir / f"{fid}_orig.jpg"
            scale = min(1.0, 960.0 / float(W))
            if scale < 1.0:
                cv2.imwrite(str(orig_dst), cv2.resize(img_bgr, (int(W * scale), int(H * scale))))
            else:
                shutil.copy(files["image"], orig_dst)

            # 2. Radar projection + GT 3D boxes
            proj = project_points_to_image(radar_raw[:, :3], calib["radar2img"], (H, W))
            proj_dst = assets_dir / f"{fid}_proj.png"
            render_radar_on_image(
                image_path=files["image"],
                uv=proj["uv"],
                depth=proj["depth"],
                calib=calib,
                annos=info.get("annos"),
                frame_id=fid,
                out_path=proj_dst,
                raw_count=proj["raw_count"],
                in_front_count=proj["in_front_count"],
            )

            # 3. BEV pred vs GT
            order = np.argsort(-p_scores)
            top_vis = [
                {
                    "score": float(p_scores[i]),
                    "label": int(p_labels[i]),
                    "name": p_names[i],
                    "box": [round(float(v), 3) for v in p_boxes[i].tolist()],
                }
                for i in order[:10]
                if float(p_scores[i]) >= args.vis_score_thr
            ]
            bev_dst = assets_dir / f"{fid}_bev.png"
            render_v2x_bev(
                data_root=data_root,
                frame_id=fid,
                dataset=dataset,
                sample_index=idx,
                predictions=top_vis,
                out_path=bev_dst,
            )

            top5 = [
                {
                    "name": p_names[i],
                    "score": float(p_scores[i]),
                    "box": [round(float(v), 3) for v in p_boxes[i].tolist()],
                }
                for i in order[:5]
            ]
            frame_entries.append(
                {
                    "frame_id": fid,
                    "gt_count": int(g_boxes.shape[0]),
                    "pred_count": int(p_boxes.shape[0]),
                    "score_max": round(float(p_scores.max()), 4) if p_scores.size else None,
                    "orig_img": f"frames/{orig_dst.name}",
                    "proj_img": f"frames/{proj_dst.name}",
                    "bev_img": f"frames/{bev_dst.name}",
                    "top5": top5,
                }
            )

    elapsed = time.perf_counter() - t0
    results_pkl = out_dir / "results.pkl"
    with open(results_pkl, "wb") as f:
        pickle.dump(raw_results, f)

    protocols = ["v2x_benchmark", "tj4d_compat"] if args.protocol == "both" else [args.protocol]
    metrics_by_protocol = {
        proto: evaluate_v2x_radar_v(gt_records, pred_records, protocol=proto, score_thr=args.score_thr)
        for proto in protocols
    }

    curve_rel = None
    curve_info = None
    if args.train_metrics:
        curve_path = showcase_dir / "training_curves.png"
        curve_info = plot_training_curves(Path(args.train_metrics), curve_path)
        if curve_info is not None:
            curve_rel = "training_curves.png"

    eval_report = {
        "status": "ok",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "ann_file": cfg.data.val.ann_file,
        "num_frames": len(dataset),
        "elapsed_seconds": round(elapsed, 3),
        "checkpoint_match": {
            "missing_keys": list(incomp.missing_keys),
            "unexpected_keys": list(incomp.unexpected_keys),
        },
        "results_pkl": str(results_pkl),
        "training_curves": curve_info,
        "metrics": metrics_by_protocol,
    }
    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(eval_report, indent=2) + "\n", encoding="utf-8")

    index_html = build_showcase_html(showcase_dir, frame_entries, eval_report, curve_rel)

    print(f"EVAL_OK frames={len(dataset)} elapsed={elapsed:.2f}s")
    for proto, m in metrics_by_protocol.items():
        print(f"  [{proto}] mAP_3D_primary={m['mAP_3d_primary_r40']} mAP_BEV_primary={m['mAP_bev_primary_r40']}")
    print(f"  metrics:  {metrics_path}")
    print(f"  showcase: {index_html}")


if __name__ == "__main__":
    main()
