# R4Det 主要概览：整体原理、运行状态与详细专题

[文档中心](README.md) · [工作区总路线](../../docs/README.md) · [单帧代码图解](../code_walkthrough/README_zh.md)

本页是 R4Det 的统一学习入口：先看输入、处理、输出的整体关系和各阶段状态，再进入原理专题。专题按“问题 → 原理/公式 → 性质 → 本地实现 → 可验证结果”展开，训练入门、代码逐段讲解和运行记录保留为配套资料。

> 文档结构与源码核对日期：2026-10-28。下面的运行状态来自已有 STATUS、trace 和运行手册，不代表本次重新运行了推理、训练或评测。配置存在、代码实现、实际执行、效果验证是四种不同证据。

## 1. R4Det 在整条雷达链上解决什么

```text
ADC → 信号处理/检测估计 → 稀疏雷达点云 ─┐
同步相机 + 标定/几何信息 ──────────────┼→ R4Det → 类别、分数、3D框
有效的相邻历史帧 ─────────────────────┘              → 评测/后续跟踪
```

它学习稀疏雷达点、图像语义、深度、时序和实例信息之间的关系。点云已经压缩了上游回波，网络可以利用上下文推断目标，但不能据此声称恢复了真实测得的弱回波或提高了雷达物理分辨率。

当前 TJ4D 管线从原始 8 列点中选择 `[x,y,z,velocity,SNR]` 五维。Radar_Process 计划输出的更丰富字段需要经过坐标、单位、特征选择与尺度适配；尚不能把这项系统接入视为已经完成。

## 2. 一帧的完整原理地图

```text
雷达点 → 0.16 m 柱网格 → RadarPillarFeatureNet → Scatter → SECOND/FPN
                                                               │原始Radar BEV
图像 → ResNet/FPN → context + 雷达辅助深度分布                    │
                     ├→ 网格painting → 与原始Radar BEV融合 ─────┤
                     └→ LSS提升/聚合 → Camera BEV ───────────────┤
                                                     Cross_Modal_Fusion
                                                               ↓
有效历史BEV ─────────────────────────────────────────→ 条件DGTF
                                                               ↓
图像特征 + 深度 + 几何 ───────────────────────────────→ LACA回投
                                                               ↓
图像RPN/RoI → 实例特征 + BEV实例图 ────────────────────→ 条件IGDR
                                                               ↓
                                                 Anchor3DHead → decode/NMS
                                                               ↓
                                                       类别、分数、3D框
```

| 部分 | 核心原理 | 需要理解的性质 | 详细解释 |
|---|---|---|---|
| 点云与 Pillar | 将变长点集聚合成固定向量，再放回平面网格 | 集合聚合、截断损失、网格与物理分辨率的区别 | [01 点云与雷达 BEV](topics/01_POINTS_PILLAR_BEV_ZH.md) |
| 图像与深度 | 像素给射线，深度给米制位置；雷达提供稀疏约束 | 深度多峰、期望值局限、几何误差传播 | [02 图像与雷达辅助深度](topics/02_IMAGE_DEPTH_ZH.md) |
| 投影与融合 | LSS 从图像分发到 BEV；painting/LACA 按空间位置读取图像 | 坐标一致性、互相调制、视觉信息进入雷达支路的位置 | [03 投影与跨模态融合](topics/03_PROJECTION_FUSION_ZH.md) |
| DGTF | 可变形采样对齐历史，门控选择候选更新 | 运动错位、历史有效性、时序调用与结果采用的区别 | [04 时序融合](topics/04_TEMPORAL_DGTF_ZH.md) |
| IGDR | 实例特征池化后投到 BEV，生成乘性/加性调制 | 实例归属、温度、空间 gate、初始化恒等性质 | [05 实例引导细化](topics/05_INSTANCE_IGDR_ZH.md) |
| 检测与训练 | anchor 残差预测、匹配监督、解码和去重 | 编码尺度、方向周期、loss 与 AP 的区别 | [06 检测头与监督](topics/06_DETECTION_TRAINING_ZH.md) |
| 验证 | 按证据层次检验接口、训练、独立数据与失效条件 | 跑通不等于效果，干预不等于正式消融 | [07 验证与能力边界](topics/07_VALIDATION_ZH.md) |

PDF、DGTF、IGDR 是整体设计中的名称；本地实现并不与论文术语一一对应。尤其 PDF 不能简单缩成最后的 `Cross_Modal_Fusion`，需要一起理解深度、投影和特征交换。

## 3. 各阶段目前是什么状态

以下 11 个阶段来自[自动状态表](../code_walkthrough/STATUS.md)及[原始 trace](../code_walkthrough/figures/r4det_trace.json)。PASS 表示保存样例的阶段执行检查通过，SKIP 表示未触发该条件路径；均不是效果评价。

| 阶段 | 保存状态 / 次数 | 输出或关键现象 | 对应原理 |
|---|---|---|---|
| image_fpn | PASS / 1 | 五层 256 通道图像特征 | 专题02 |
| depth_distribution | PASS / 1 | context `[1,1,256,60,80]`，depth `[1,72,60,80]` | 专题02 |
| camera_bev | PASS / 1 | 轴重排前 `[1,256,216,248,1]` | 专题03 |
| radar_pillars | PASS / 1 | `[423,10,5] → [423,64]` | 专题01 |
| radar_scatter | PASS / 1 | `[1,64,496,432]` | 专题01 |
| radar_bev | PASS / 1 | `[1,384,248,216]` | 专题01 |
| cross_modal_fusion | PASS / 1 | `[1,256,248,216]` | 专题03 |
| temporal_fusion | **SKIP / 0** | 样例没有有效历史，未验证时序计算 | 专题04 |
| backward_projection | PASS / 1 | `[1,53568,256]`，53568=216×248 | 专题03 |
| instance_guided_refinement | PASS / 1 | 输入两个实例，输出 `[1,256,248,216]` | 专题05 |
| detection_head | PASS / 1 | 分类32、框56、方向16通道 | 专题06 |

表中 shape 仅适用于已保存样例，不是所有配置的固定常数。painting 等内部步骤并没有在该状态表中独立列项，不能为它们凭空补造独立 PASS。该次记录总状态 PASS、候选框 300；候选数量不等于真实目标数量。

## 4. 工程整体完成到哪一层

| 范围 | 已有证据 | 仍需完成 |
|---|---|---|
| 环境与单帧 | 运行手册记录 RTX 4060 Ti FP32/FP16 smoke，完整 checkpoint 严格加载 | 换环境后按手册重验 |
| 数据 | 样例有真实 TJ4D 雷达、标定和标签 | 原始同步相机与完整正式 split；公开 smoke 是雷达投影代理图 |
| 时序 | 有实现、配置和条件接口 | 真实有效相邻帧验证；保存 trace 中未触发 |
| 实例分支 | 保存样例执行一次 | 真实实例正确性与检测收益对照 |
| 训练 | 有训练入口和监督实现 | L2 小数据过拟合、梯度/更新及恢复训练证据 |
| 正式指标 | 当前资料没有足够完成证据 | L3 完整评测及 RadarOnly/CameraOnly/Fusion 对照 |
| 推理合同 | 当前接口依赖多项几何、历史和 GT 元数据 | 无标签输入适配、GT 依赖审计 |
| VoD | 有相关工具与训练入口名称 | 本地上游配置不完整，不能视为一键复现就绪 |
| 系统接入与部署 | 已有点合同和实验规划 | Radar_Process 适配、失效降级、真实跟踪输出及性能验证 |

运行环境、命令、权重校验和历史性能统一查[运行手册](R4DET_RUNBOOK.md)；后续消融与接入清单查[点云融合实验路线](POINT_CLOUD_FUSION_LEARNING_ZH.md)。本页仅汇总状态含义，不复制运行命令和性能数字。

## 5. 训练、配置、模型和权重的关系

```text
配置 → Dataset/Pipeline → DataLoader → 一批输入与标注
配置 → 模型结构 ─────────────────────→ 前向预测与loss
优化器/runner ──────────────────────→ 反向传播、参数更新、保存checkpoint
新输入 + 同结构模型 + checkpoint ───→ 推理与后处理
```

配置决定选什么，源码规定怎样算，运行时对象持有模型参数，checkpoint 保存训练状态。训练不会自动把权重写回 Python 源码。

当前 [TJ4D 配置](../configs/r4det/TJ4D-R4Det_det3d_2x4_12e.py) 为 `TJ4DDataset` 外包 `RepeatDataset(times=2)`，每 GPU 4 样本，AdamW 初始学习率 0.0002，`runner.max_epochs=18`。文件名中的 `12e` 不决定实际轮数。`load_from` 指向 `epoch_2.pth`，`resume_from=None`；它表达权重起点，不表示完整恢复上次训练进度。命令行覆盖后以实际生效配置为准。

## 6. 推荐阅读顺序与配套资料

先读本页的整体链与状态，再按原理专题 01 → 07 阅读。完全不了解训练时，先补前三项入门资料；需要逐行对应代码时再查单帧图解。

| 目的 | 资料 |
|---|---|
| 理解样本、loss、梯度、epoch | [深度学习训练基础](TRAINING_BASICS_ZH.md) |
| 理解注册、Dataset、Pipeline、batch | [配置与数据流](CONFIG_AND_DATA_ZH.md) |
| 跟踪脚本到框架的控制权转交 | [train_vod.py 详解](../code_walkthrough/TRAIN_VOD_EXPLAINED_ZH.md) |
| 把专题原理对应到每次函数调用 | [单帧模型详解](../code_walkthrough/R4Det_ONE_FRAME_EXPLAINED_ZH.md) |
| 理解保存、加载和恢复 | [权重与推理](CHECKPOINT_AND_INFERENCE_ZH.md) |
| 追溯旧综合笔记 | [工作区 R4Det 综合专题](../../docs/topics/02_R4DET_ZH.md) |

读完应能手算一个柱索引，解释深度分布与两种投影方向，区分 DGTF/IGDR，解释 32/56/16 输出通道，并准确指出代理相机、时序 SKIP、训练/正式指标待验证三类边界。之后即可继续工作区的 DinoRADE 学习路线。
