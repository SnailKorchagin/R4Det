# R4Det × V2X-Radar-V 实施手册

[学习总览](LEARNING_OVERVIEW_ZH.md) · [文档中心](README.md) · [TJ4D 基线运行手册](R4DET_RUNBOOK.md)

## 给 Gemini 的开工指令

> 在 `/home/snailkorchagin/Radar/R4Det` 工作。直接实现并验证 **V2X-Radar-V 到 R4Det 的数据集适配**，覆盖阶段 1 和阶段 2。交付数据读取与转换代码、可由 R4Det 配置构建的 Dataset、独立 V2X 配置和真实单帧验收入口。投影图是几何校验，适配须在 R4Det 的正常数据加载入口可用。优先复用已有数据和仓库逻辑。若缺数据或 CUDA，继续完成可独立验证的解析、转换、注册和配置，分别报告真实样本读取、batch 构建和 GPU 前向的验证状态。交付时写明改动文件、运行命令、产物路径、实际帧号和阻塞。

## 起点与交付目标

主线是 **在 V2X-Radar-V 车端真实雷达与相机数据上移植 R4Det，完成训练、验证集评测和逐帧展示**。当前工作区的实测证据来自 TJ4D `070070`：官方权重严格加载，FP32/FP16 单帧通过，视觉输入是雷达投影代理图，DGTF 在保存的 trace 中为 `SKIP`。V2X 数据解析、真实投影、推理、训练与 AP 均待运行验证。最终指标以 V2X-Radar-V 的数据划分和评测协议报告。 [冻结基线](../../baselines/2026-09-15/README.md) · [现有 trace 状态](../code_walkthrough/STATUS.md) · [V2X-Radar 论文](https://arxiv.org/html/2411.10962v2)

## 1. 数据与运行环境

- 从车端目录获取 `V2X-Radar-V.zip`（约 23.9 GB）和 `ImageSets.zip`。先查本地是否已有归档或解压目录，仅补缺件。归档可放在 `.downloads/`，解压数据可放在 `data/V2X-Radar-V/` 或通过命令行指向已有数据根。用 `unzip -l` 核对实际目录层级与文件名，再确定适配路径。车端数据采用 KITTI 风格组织，点字段以真实文件和官方定义为准。 [官方车端文件目录](https://huggingface.co/datasets/yanglei18/V2X-Radar/tree/main/V2X-Radar-V) · [官方数据布局](https://github.com/yanglei18/V2X-Radar#data-download)
- 记录来源、发布版本或文件哈希、使用的划分与帧号。许可为 CC BY-NC-ND 4.0。原始数据保存在本地，公开代码或展示产物前核对数据许可。 [官方数据卡与许可](https://huggingface.co/datasets/yanglei18/V2X-Radar)
- 先复用 `.venv`。数据检查用 Python、NumPy 和图像库即可。模型前向前用 `.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'` 检查 CUDA。已有环境是 PyTorch 2.0.1+cu118、MMCV 1.7.2，原生算子编译面向 RTX 4060 Ti 的 `sm_89`。以实际运行结果记录 GPU 和算子状态。 [环境细节](R4DET_RUNBOOK.md)

## 2. 现有源码地图

| 任务 | 先读哪里 | 实施时检查什么 |
|---|---|---|
| 数据读取与标定 | [TJ4D 检测配置](../configs/r4det/TJ4D-R4Det_det3d_2x4_12e.py)、[TJ4DDataset](../mmdet3d/datasets/TJ4D_dataset.py)、[VoD 数据转换](../tools/data_converter/kitti_data_utils_VOD.py) | 参考现有数据结构。按 V2X 文件独立核对二进制字段、雷达到相机矩阵、深度坐标和标签坐标。 |
| 单帧、权重与输出 | [单帧入口](../tools/r4det_smoke_test.py)、[模型源码](../mmdet3d/models/detectors/R4Det.py)、[trace 与 BEV 渲染](../code_walkthrough/generate_r4det_storyboard.py) | 跟 `simple_test → extract_feat → 3D head → decode/NMS`，复用可独立调用的预测整理、hook 和绘图逻辑。 |
| 训练 | [训练 CLI](../tools/train_vod.py)、[训练 API](../mmdet3d/apis/train_vod.py)、[启动脚本](../tools/dist_train.sh) | 跟 Dataset、DataLoader、`forward_train`、loss、优化器和 checkpoint。确认标注、实例 mask 和深度监督分别由哪里读取或生成。 |
| 评测 | [现有测试 CLI](../tools/test_vod.py)、[V2X 官方实现](https://github.com/yanglei18/V2X-Radar)、[论文车端基准](https://arxiv.org/html/2411.10962v2) | 现有训练和测试脚本含按配置名复制评测源码的逻辑。V2X 入口应显式选择评测器，避免运行时覆盖仓库源码。 |

### 数据集适配的必交代码

1. **统一读取与转换。** 接收数据根和官方 split，按原始帧 ID 联合读取雷达、图像、标定与标签。数据检查工具和 R4Det Dataset 共用同一套配对与几何逻辑。用真实文件确认记录格式、点字段、标定矩阵方向、标签坐标、框原点和尺寸顺序。保留原始类别及帧 ID，明确训练类别映射和忽略规则。对标注缺失和合法空标注分别处理。
2. **Dataset 与配置。** 通过现有注册机制让 `build_dataset` 能用独立 V2X 配置构建训练和验证数据集，`build_dataloader` 能取出样本。复用适用的既有 Dataset 和 pipeline。按实际模型输入核对点、图像、增强后几何和元数据；有标注的训练路径应能产生 3D 框和类别，无标签路径应能读取传感器输入。记录 V2X `intensity` 到模型第五列的处理方式。遇到缺失的深度监督、实例 mask 或推理所需字段，定位消费方并实现有依据的处理或报告具体缺件。单帧输入显式标记无有效历史。
3. **真实样本验收。** 用同一真实帧输出投影图和样本摘要。通过实际 V2X 配置读取一个示例 batch，说明字段、shape、坐标、类别和标签数量，并检查至少一个带框样本、合法空框和关键输入缺失的行为。有 CUDA 时用该 batch 跑单帧前向，保存预测和 BEV 图。随代码交付带真实帧号的命令及可重复的标定、字段转换检查。

## 3. 实施阶段与通过条件

| 阶段 | 实施任务 | 通过条件 |
|---|---|---|
| **1. CPU 数据审计与投影** | 做独立 CLI。按帧 ID 配对雷达、图像、标定和标签，检查官方划分成员及交集。以真实标定计算投影，筛掉相机后方、非法坐标和画面外的点。分别检查序列身份、帧序和时间字段的可用性。 | 真实样例产出 `inspection.json` 和 `radar_on_image.png`。报告包含实际使用的标定、帧 ID、原始/有效/入画点数、图像大小、缺失文件和检查结果。校验至少包含一个已知几何点的投影，以及画外或负深度点的过滤。 |
| **2. 真实单帧 GPU 前向** | 新建 V2X 数据适配与独立配置，先跑单帧。标明 `intensity` 的定义和尺度选择，核对增强后的投影与 BEV 轴。审计 `simple_test` 消费的 GT 与辅助输入，用固定权重和图像/雷达，移除或扰动 GT 比较预测。 | 生成预测 JSON、BEV 图、输入来源和阶段 trace。记录权重匹配情况及是否用到 GT。TJ4D 权重在此只用于跨数据集前向检查。无法移除的 GT 依赖应记录具体调用处及实际影响。 |
| **3. 小样本训练与恢复** | 清点训练管线要求的标注、实例掩码和深度监督，生成能复跑的真实数据预处理。有依据地确定单帧配置与初始化策略，检查单 GPU 的 batch 和显存。 | 一组固定样本的损失出现可解释的下降，关键梯度有限且参数发生更新。保存 checkpoint，再恢复并确认迭代/epoch、优化器状态和学习率能接上。产出配置和日志。 |
| **4. 全量训练与时序验证** | 锁定车端官方 split、类别映射、初始化、随机种子和有效配置，在完整训练集上跑单帧基线。若确认能可靠识别同序列相邻帧及有效间隔，再增加 DGTF 配置。 | 单帧基线有完整日志和可加载权重。多帧路径同时检查有/无历史、调用次数和时序输出是否实际进入下游。缺可靠序列信息时仅报告单帧结果，并记录时序阻塞。 |
| **5. 评测与展示** | 选择并记录官方车端划分和协议，确认 Vehicle、Pedestrian、Cyclist 映射、IoU 阈值、难度与 AP 口径。验证结果导出和评测器输入一致。生成本地逐帧结果页和训练曲线。 | 一条评测命令可从指定 checkpoint 重跑。交付类别 AP、实际划分、配置和权重路径。结果页能切换帧，同时看到原图、雷达投影、BEV 预测与 GT；训练曲线对应同一次实验。 |

正式评测前检查 train/val 帧 ID 重叠。如果序列信息可用，也检查两边是否共享同一段序列。保留对官方划分的复测口径；若补做按序列留出的泛化实验，单独记录其划分和结果。 [车端官方协议](https://arxiv.org/html/2411.10962v2)

## 4. 阶段 1 的代码与验收命令

拟新增 `tools/inspect_v2x_radar_v.py`。下面是实现目标，当前工作区还没有这个入口。CLI 最少接收数据根、帧 ID 和输出目录。全量检查只输出摘要与异常项；单帧输出投影图。实现时优先复用已有可独立调用的解析和绘图代码；如果现有模块在导入时要求 CUDA，阶段 1 使用 NumPy/图像库的 CPU 路径。

```bash
cd /home/snailkorchagin/Radar/R4Det
.venv/bin/python tools/inspect_v2x_radar_v.py \
  --data-root data/V2X-Radar-V \
  --frame-id "$V2X_FRAME_ID" \
  --out-dir work_dirs/v2x_radar_v/inspection
```

`V2X_FRAME_ID` 使用目录中实际存在、四种文件均齐备的帧 ID。阶段 1 交付时，把占位变量换成真实 ID，说明 `--data-root` 指向解压目录的哪一层。如果数据尚未下载完，报告 CPU 工具与几何校验结果；真实投影仍标为待验证。

后续的新配置建议放进 `configs/r4det/`，用 `V2X-Radar-V` 前缀。训练、推理和评测优先复用现有框架入口，按实际需要补适配。权重、预测、日志和本地展示输出放 `work_dirs/v2x_radar_v/`。现有 TJ4D smoke 和图解是回归参照，V2X 代码接入前后应检查其入口仍可用。 [TJ4D 复跑命令](R4DET_RUNBOOK.md)

## 5. 学习与实施怎么对上

每做完一阶段，留一个可用的讲解样本。阶段 1 用真实帧读点字段和标定；阶段 2 用同一帧的 trace 读 Pillar、深度、BEV、IGDR 和检测头；阶段 3 用真实训练日志解释 loss、梯度和 checkpoint；阶段 4 用两帧的实际对照解释 DGTF；阶段 5 用预测和 GT 解释 AP、误检和漏检。原理阅读入口见[学习总览的实施顺序](LEARNING_OVERVIEW_ZH.md#6-跟着工程产物学习)。
