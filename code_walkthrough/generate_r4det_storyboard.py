#!/usr/bin/env python3
"""Trace TJ4D frame 070070 through R4Det and render learning figures.

Forward hooks capture the actual weighted inference path without changing model
code. Conditional paths that do not execute are reported as SKIP, never PASS.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/r4det-walkthrough-mpl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from tools.r4det_smoke_test import unwrap_prediction


FRAME_ID = "070070"
OUT = ROOT / "code_walkthrough" / "figures"
TRACE_PATH = OUT / "r4det_trace.json"
STATUS_PATH = ROOT / "code_walkthrough" / "STATUS.md"
PC_RANGE = [0.0, -39.68, -4.0, 69.12, 39.68, 2.0]
BEV_H, BEV_W = 216, 248


def tensor_leaves(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return []
    leaves: list[torch.Tensor] = []
    for child in children:
        leaves.extend(tensor_leaves(child))
    return leaves


def finite_stats(tensor: torch.Tensor) -> dict[str, Any]:
    x = tensor.detach().float()
    finite = torch.isfinite(x)
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "finite": bool(finite.all()),
    }
    if x.numel() and finite.any():
        y = x[finite]
        result.update(min=round(float(y.min()), 6), max=round(float(y.max()), 6), mean=round(float(y.mean()), 6))
    return result


def summarize(value: Any) -> list[dict[str, Any]]:
    return [finite_stats(item) for item in tensor_leaves(value)]


def energy_map(value: Any, name: str) -> np.ndarray | None:
    leaves = tensor_leaves(value)
    if name == "depth_distribution":
        for item in leaves:
            if item.ndim == 4 and item.shape[1] == 72:
                probs = item.detach().float().cpu()[0]
                depths = torch.arange(1.0, 73.0).view(-1, 1, 1)
                return (probs * depths).sum(0).numpy()
    if name == "detection_head":
        for item in leaves:
            if item.ndim == 4 and item.shape[1] >= 4:
                # Head layout is [B,C,y,x]; figures use rows=x, cols=y.
                return item.detach().float().sigmoid().amax(1)[0].cpu().numpy().T
    if name == "camera_bev":
        for item in leaves:
            if item.ndim == 5 and item.shape[2:4] == (BEV_H, BEV_W):
                # ViewTransformerLSS is [B,C,x,y,z].
                return item.detach().float().abs().mean((0, 1, 4)).cpu().numpy()
    if name == "backward_projection":
        for item in leaves:
            if item.ndim == 3 and item.shape[1] == BEV_H * BEV_W:
                # BackwardProjection is [B,x*y,C].
                return item.detach().float()[0].abs().mean(-1).reshape(BEV_H, BEV_W).cpu().numpy()
    if name in {"radar_bev", "cross_modal_fusion", "instance_guided_refinement"}:
        for item in leaves:
            if item.ndim == 4 and item.shape[2:4] == (BEV_W, BEV_H):
                # These modules are [B,C,y,x]; figures use rows=x, cols=y.
                return item.detach().float()[0].abs().mean(0).cpu().numpy().T
    candidates = [item for item in leaves if item.ndim >= 3]
    if not candidates:
        return None
    tensor = max(candidates, key=lambda item: item.numel()).detach().float().cpu()
    if tensor.ndim == 5:
        return tensor.abs().mean(tuple(range(tensor.ndim - 2))).numpy()
    if tensor.ndim == 4:
        return tensor[0].abs().mean(0).numpy()
    if tensor.ndim == 3 and tensor.shape[0] == BEV_H * BEV_W:
        return tensor[:, 0, :].abs().mean(1).reshape(BEV_H, BEV_W).numpy()
    if tensor.ndim == 3:
        return tensor.abs().mean(0).numpy()
    return None


class TraceRecorder:
    def __init__(self) -> None:
        self.calls: dict[str, list[dict[str, Any]]] = {}
        self.maps: dict[str, np.ndarray] = {}
        self.handles = []

    def hook(self, name: str):
        def capture(_module, inputs, output):
            self.calls.setdefault(name, []).append({"input": summarize(inputs), "output": summarize(output)})
            mapped = energy_map(output, name)
            if mapped is not None:
                self.maps[name] = mapped
        return capture

    def register(self, model) -> None:
        modules = {
            "image_fpn": model.img_neck,
            "depth_distribution": model.depth_net,
            "camera_bev": model.img_view_transformer,
            "radar_pillars": model.pts_voxel_encoder,
            "radar_scatter": model.pts_middle_encoder,
            "radar_bev": model.pts_neck,
            "cross_modal_fusion": model.cross_attention,
            "temporal_fusion": model.temporal_fusion,
            "backward_projection": model.backward_projection,
            "instance_guided_refinement": model.ifgd_fusion,
            "detection_head": model.pts_bbox_head,
        }
        for name, module in modules.items():
            if module is not None:
                self.handles.append(module.register_forward_hook(self.hook(name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def normalize(image: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(image, dtype=np.float32))
    low, high = np.percentile(array, [2, 98])
    return np.clip((array - low) / max(float(high - low), 1e-8), 0.0, 1.0)


def resize_map(image: np.ndarray, shape=(BEV_H, BEV_W)) -> np.ndarray:
    return cv2.resize(normalize(image), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def draw_box(ax, box, color, label=None, linewidth=2.0):
    x, y, _z, dx, dy, _dz, yaw = [float(v) for v in box[:7]]
    local = np.array([[dx / 2, dy / 2], [dx / 2, -dy / 2], [-dx / 2, -dy / 2], [-dx / 2, dy / 2]])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    corners = local @ rotation.T + np.array([x, y])
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=linewidth))
    if label:
        ax.text(x, y, label, color=color, fontsize=8, weight="bold")


def plot_raw_bev(ax, radar: np.ndarray, gt_boxes: np.ndarray, predictions: list[dict]):
    ax.scatter(radar[:, 1], radar[:, 0], s=2, c=radar[:, 3], cmap="coolwarm", alpha=0.65)
    for index, box in enumerate(gt_boxes):
        swapped = [box[1], box[0], box[2], box[4], box[3], box[5], -box[6]]
        draw_box(ax, swapped, "#ff4f64", "GT" if index == 0 else None)
    for prediction in predictions[:10]:
        box = prediction["box"]
        swapped = [box[1], box[0], box[2], box[4], box[3], box[5], -box[6]]
        draw_box(ax, swapped, "#53e07c", None, 1.0)
    ax.set(xlim=(PC_RANGE[1], PC_RANGE[4]), ylim=(PC_RANGE[0], PC_RANGE[3]), xlabel="y lateral (m)", ylabel="x forward (m)")
    ax.set_aspect("equal")


def plot_feature(ax, image: np.ndarray | None, title: str, bev=False, cmap="magma"):
    if image is None:
        ax.text(0.5, 0.5, "SKIP / not captured", ha="center", va="center", color="#ffb454", transform=ax.transAxes)
        ax.set_facecolor("#20242c")
    elif bev:
        ax.imshow(normalize(image), origin="lower", extent=[PC_RANGE[1], PC_RANGE[4], PC_RANGE[0], PC_RANGE[3]], aspect="auto", cmap=cmap)
        ax.set(xlabel="y (m)", ylabel="x (m)")
    else:
        ax.imshow(normalize(image), cmap=cmap, aspect="auto")
        ax.set(xlabel="feature col", ylabel="feature row")
    ax.set_title(title, fontsize=10)


def render_overview(stage_records: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")
    lanes = [
        ("Radar lane", 0.73, "#ffb454", ["raw radar\n[N,5]", "pillars + scatter", "SECOND + FPN\nradar BEV"]),
        ("Camera lane", 0.48, "#62b0ff", ["proxy RGB\n[3,480,640]", "ResNet50 + FPN", "depth + LSS\ncamera BEV"]),
        ("Fusion lane", 0.20, "#7ee787", ["cross-modal\nattention", "LACA backward\nprojection", "IGDR + 3D head\nboxes + scores"]),
    ]
    for lane_name, y, color, boxes in lanes:
        ax.text(0.02, y + 0.05, lane_name, color=color, fontsize=13, weight="bold")
        for x, label in zip([0.18, 0.48, 0.78], boxes):
            rect = FancyBboxPatch((x, y), 0.18, 0.11, boxstyle="round,pad=0.015", facecolor="#161b22", edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x + 0.09, y + 0.055, label, ha="center", va="center", color="white", fontsize=11)
        for x in (0.37, 0.67):
            ax.annotate("", xy=(x + 0.09, y + 0.055), xytext=(x, y + 0.055), arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax.annotate("", xy=(0.24, 0.31), xytext=(0.24, 0.48), arrowprops=dict(arrowstyle="->", color="#8b949e", lw=2))
    ax.annotate("", xy=(0.30, 0.31), xytext=(0.30, 0.73), arrowprops=dict(arrowstyle="->", color="#8b949e", lw=2))
    passed = sum(item["status"] == "PASS" for item in stage_records)
    skipped = sum(item["status"] == "SKIP" for item in stage_records)
    ax.text(0.02, 0.94, "R4Det frame 070070 — actual weighted forward trace", color="white", fontsize=21, weight="bold")
    ax.text(0.02, 0.89, f"{passed} executed stages PASS · {skipped} conditional stages SKIP", color="#c9d1d9", fontsize=12)
    ax.text(0.02, 0.05, "Public proxy camera validates code/data contracts, not published AP.", color="#ffb454", fontsize=11)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_process(recorder: TraceRecorder, proxy: np.ndarray, radar: np.ndarray, gt_boxes: np.ndarray, predictions: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
    axes[0, 0].imshow(cv2.cvtColor(proxy, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("1 Proxy camera (not original RGB)"); axes[0, 0].axis("off")
    plot_raw_bev(axes[0, 1], radar, gt_boxes, []); axes[0, 1].set_title("2 Raw radar points + GT")
    plot_feature(axes[0, 2], recorder.maps.get("image_fpn"), "3 Image FPN energy")
    plot_feature(axes[0, 3], recorder.maps.get("depth_distribution"), "4 Expected depth (m)", cmap="viridis")
    plot_feature(axes[1, 0], recorder.maps.get("radar_scatter"), "5 Pillar scatter", cmap="cividis")
    plot_feature(axes[1, 1], recorder.maps.get("radar_bev"), "6 Radar BEV", bev=True, cmap="cividis")
    plot_feature(axes[1, 2], recorder.maps.get("camera_bev"), "7 Camera LSS BEV", bev=True, cmap="Blues")
    plot_feature(axes[1, 3], recorder.maps.get("cross_modal_fusion"), "8 Cross-modal BEV", bev=True)
    plot_feature(axes[2, 0], recorder.maps.get("backward_projection"), "9 LACA refinement", bev=True)
    plot_feature(axes[2, 1], recorder.maps.get("instance_guided_refinement"), "10 IGDR refinement", bev=True)
    plot_feature(axes[2, 2], recorder.maps.get("detection_head"), "11 Head objectness", bev=True)
    plot_raw_bev(axes[2, 3], radar, gt_boxes, predictions); axes[2, 3].set_title("12 GT red / predictions green")
    fig.suptitle("R4Det: one frame from sensor inputs to 3D candidates", fontsize=19, weight="bold")
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_fusion_closeup(recorder: TraceRecorder, gt_boxes: np.ndarray, output: Path) -> None:
    names = ["radar_bev", "camera_bev", "cross_modal_fusion", "backward_projection", "instance_guided_refinement"]
    maps = {name: resize_map(recorder.maps[name]) for name in names if name in recorder.maps}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    titles = ["Radar BEV", "Camera BEV", "Cross-modal fusion", "LACA refined", "IGDR refined"]
    for ax, name, title in zip(axes.flat[:5], names, titles):
        plot_feature(ax, maps.get(name), title, bev=True)
        if len(gt_boxes):
            ax.scatter([gt_boxes[0, 1]], [gt_boxes[0, 0]], s=90, facecolors="none", edgecolors="#00e5ff", linewidths=2)
    if "radar_bev" in maps and "backward_projection" in maps:
        plot_feature(axes.flat[5], np.abs(maps["backward_projection"] - maps["radar_bev"]), "|LACA − radar| change", bev=True, cmap="inferno")
    else:
        plot_feature(axes.flat[5], None, "Change map")
    for ax in axes.flat:
        ax.set_xlim(-20, 20); ax.set_ylim(0, 65)
    fig.suptitle("R4Det BEV feature evolution — same metric window", fontsize=19, weight="bold")
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run through run_r4det_walkthrough.sh")
    OUT.mkdir(parents=True, exist_ok=True)
    logging.getLogger("mmcv").setLevel(logging.WARNING)
    #该配置中包含模型结构和数据集，数据预处理和测试参数
    cfg = Config.fromfile(str(ROOT / "configs/r4det/TJ4D-R4Det_det3d_2x4_12e.py"))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.meta_info = {"figures_path": str(ROOT / ".smoke/figures"), "project_name": "TJ4D-R4Det-walkthrough"}
    cfg.data.test.ann_file = str(ROOT / "data/TJ4D/TJ4D_infos_public_smoke.pkl")
    cfg.data.test.data_root = str(ROOT / "data/TJ4D") + "/"
    cfg.data.test.test_mode = False
    #创建数据集对象
    #pipeline:雷达读取
    # → 图像读取
    # → 读取分割结果
    # → 读取 2D/3D 标注
    # → 图像增强
    # → 点云范围过滤
    # → 图像归一化
    # → 图像 padding
    # → 生成雷达深度
    # → 转换为 tensor/DataContainer
    # → 收集模型需要的字段
    dataset = build_dataset(cfg.data.test)
    #创建batch读取数据迭代器，每一轮调用dataset
    data = next(iter(build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)))

    # build_model(cfg.model)
    # └── 创建 R4Det
    #     ├── 创建图像 backbone
    #     ├── 创建图像 FPN
    #     ├── 创建深度网络
    #     ├── 创建 LSS view transformer
    #     ├── 创建 radar voxel encoder
    #     ├── 创建 radar middle encoder
    #     ├── 创建 SECOND backbone
    #     ├── 创建 radar FPN
    #     ├── 创建跨模态融合模块
    #     ├── 创建 backward projection
    #     └── 创建 3D detection head
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, str(ROOT / "checkpoints/final_tj4d.pth"), map_location="cpu")
    # 1. 准备模型并注册监听器
    # 2. 执行一次真实前向推理
    # 3. 检查每个网络模块是否正常执行
    # 4. 读取可视化需要的原始数据并画图
    # 5. 保存 JSON、Markdown 报告并打印结果
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    recorder = TraceRecorder(); recorder.register(model)
    parallel_model = MMDataParallel(model.cuda().eval(), device_ids=[0])
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    #这里只执行推理，不记录梯度，减少显存占用
    with torch.inference_mode():
        #执行前向传播
        #前面准备的是创建模型，加载权重，注册hook，移动到GPU
        output = parallel_model(return_loss=False, rescale=True, **data)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started; recorder.close()
    prediction = unwrap_prediction(output)

    required = [
        ("image_fpn", "proxy RGB → ResNet50/FPN"), ("depth_distribution", "radar-assisted depth"),
        ("camera_bev", "LSS image-to-BEV"), ("radar_pillars", "radar point → pillar"),
        ("radar_scatter", "pillar → dense scatter"), ("radar_bev", "SECOND/SECONDFPN"),
        ("cross_modal_fusion", "radar-camera BEV fusion"), ("temporal_fusion", "previous/current BEV"),
        ("backward_projection", "LACA refinement"), ("instance_guided_refinement", "IGDR refinement"),
        ("detection_head", "3D anchor head"),
    ]
    conditional = {"temporal_fusion", "instance_guided_refinement"}
    stage_records = []
    for name, purpose in required:
        calls = recorder.calls.get(name, [])
        outputs_finite = bool(calls) and all(item["finite"] for item in calls[-1]["output"])
        if calls:
            status = "PASS" if outputs_finite else "FAIL"
            reason = None if outputs_finite else "one or more output tensors contain NaN/Inf"
        else:
            status = "SKIP" if name in conditional else "FAIL"
            reason = "conditional path had no valid previous frame or 2D instance" if name in conditional else "required module was not called"
        stage_records.append({
            "name": name, "purpose": purpose,
            "status": status,
            "call_count": len(calls), "last_call": calls[-1] if calls else None,
            "reason": reason,
        })
    failed = [stage["name"] for stage in stage_records if stage["status"] == "FAIL"]
    if failed:
        raise RuntimeError(f"required trace stages were not captured: {failed}")

    radar = np.fromfile(ROOT / f"data/TJ4D/training/velodyne_reduced/{FRAME_ID}.bin", dtype=np.float32).reshape(-1, 8)
    proxy_path = ROOT / f"data/TJ4D/training/image_2/{FRAME_ID}.png"
    proxy = cv2.imread(str(proxy_path), cv2.IMREAD_COLOR)
    if proxy is None:
        raise RuntimeError(f"cannot read {proxy_path}")
    gt_boxes = dataset.get_ann_info(0)["gt_bboxes_3d"].tensor.cpu().numpy()
    render_overview(stage_records, OUT / "01_r4det_pipeline_overview.png")
    render_process(recorder, proxy, radar, gt_boxes, prediction["top10"], OUT / "02_frame_070070_process.png")
    render_fusion_closeup(recorder, gt_boxes, OUT / "03_bev_fusion_evolution.png")

    trace = {
        "frame_id": FRAME_ID, "status": "PASS",
        "input": {"raw_radar_shape": list(radar.shape), "model_radar_features": "x,y,z,velocity,SNR", "camera_file": str(proxy_path), "camera_is_original": False, "camera_shape": list(proxy.shape), "ground_truth_boxes": int(len(gt_boxes))},
        "environment": {"gpu": torch.cuda.get_device_name(0), "pytorch": torch.__version__, "elapsed_seconds": round(elapsed, 4), "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1)},
        "stages": stage_records, "output": prediction,
        "figures": ["01_r4det_pipeline_overview.png", "02_frame_070070_process.png", "03_bev_fusion_evolution.png"],
        "valid_for_paper_metric_reproduction": False,
    }
    TRACE_PATH.write_text(json.dumps(trace, indent=2) + "\n")
    rows = []
    for index, stage in enumerate(stage_records, 1):
        last = stage["last_call"]
        input_shapes = [item["shape"] for item in last["input"]] if last else []
        output_shapes = [item["shape"] for item in last["output"]] if last else []
        rows.append(f"| {index} | `{stage['name']}` | {stage['status']} | `{input_shapes}` | `{output_shapes}` | {stage['call_count']} |")
    STATUS_PATH.write_text(
        "# R4Det 单帧过程状态（自动生成）\n\n> 生成命令：`./run_r4det_walkthrough.sh`。详细数值见 `figures/r4det_trace.json`。\n\n"
        "| # | 阶段 | 状态 | 输入 shape | 输出 shape | 调用次数 |\n|---:|---|---|---|---|---:|\n" + "\n".join(rows)
        + f"\n\n总状态：**PASS**；候选框：{prediction['detections']}；耗时：{elapsed:.4f} s；峰值显存：{trace['environment']['peak_gpu_memory_mib']} MiB。\n\n"
        "`SKIP` 只允许出现在有明确运行条件的 temporal/IGDR 路径；它表示本样例没有触发，不等于模块通过效果验证。\n"
    )
    print("R4DET_WALKTHROUGH_OK")
    print(f"  stages: {sum(s['status'] == 'PASS' for s in stage_records)} PASS, {sum(s['status'] == 'SKIP' for s in stage_records)} SKIP")
    print(f"  detections: {prediction['detections']}")
    print(f"  elapsed: {elapsed:.4f} s, peak CUDA: {trace['environment']['peak_gpu_memory_mib']} MiB")
    print(f"  output: {OUT}")


if __name__ == "__main__":
    main()
