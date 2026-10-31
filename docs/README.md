# R4Det 文档中心

> 学习总顺序以[工作区总路线](../../docs/README.md)为准；R4Det 的整体原理、阶段状态和专题阅读顺序统一从[主要概览](LEARNING_OVERVIEW_ZH.md)进入。

> 本工程负责稀疏4D雷达点云、Pillar/voxel BEV、雷达辅助深度、点云—相机融合、时序 BEV 和实例引导净化。

[返回工作区总路线](../../docs/README.md)

<a id="top"></a>

## 目录

- [1. 工程定位](#scope)
- [2. 当前复现状态](#status)
- [3. 详细文档](#documents)
- [4. 推荐执行顺序](#order)
- [5. 完成标准](#done)

<a id="scope"></a>

## 1. 工程定位

```text
Radar points → voxel/pillar → Radar BEV
RGB → backbone/FPN → radar-assisted depth → Camera BEV
Radar BEV + Camera BEV
→ PDF/cross-modal fusion
→ DGTF temporal fusion
→ IGDR instance refinement
→ 3D detection
```

计划通过适配器接收 Radar_Process 输出的丰富点合同；下面是目标字段示意，当前 TJ4D 模型实际选择五维点输入，系统接入尚待验证：

```text
[x,y,z,radial_velocity,power,SNR,covariance,source]
```

[返回目录](#top)

<a id="status"></a>

## 2. 当前复现状态

逐阶段 PASS/SKIP、训练/数据/部署待办见[主要概览状态表](LEARNING_OVERVIEW_ZH.md#3-各阶段目前是什么状态)。以下为已有运行记录摘要，并非本次重跑结果。

已验证：

- RTX 4060 Ti上的FP32/FP16单帧推理；
- 作者TJ4D完整checkpoint严格加载；
- 真实TJ4D雷达、标定和标签；
- 雷达/视觉/深度/BEV/融合/检测/NMS 单帧链路；保存 trace 中 DGTF 为 SKIP，IGDR 执行过。

边界：

- 当前公开 smoke 的视觉输入是雷达投影代理图；
- 结果只证明软件链可运行，不能复现论文 AP；
- 正式验证需要完整同步相机和官方 split；
- 本地上游 VoD 配置不完整，不能假定已具备一键 VoD 训练能力。

[返回目录](#top)

<a id="documents"></a>

## 3. 详细文档

- [主要概览：整体原理、全部阶段状态与阅读路线](LEARNING_OVERVIEW_ZH.md)
- [原理01：点云、Pillar 与雷达 BEV](topics/01_POINTS_PILLAR_BEV_ZH.md)
- [原理02：图像与雷达辅助深度](topics/02_IMAGE_DEPTH_ZH.md)
- [原理03：LSS、painting、跨模态融合与 LACA](topics/03_PROJECTION_FUSION_ZH.md)
- [原理04：DGTF 时序对齐与门控](topics/04_TEMPORAL_DGTF_ZH.md)
- [原理05：IGDR 实例分配与条件调制](topics/05_INSTANCE_IGDR_ZH.md)
- [原理06：检测头、损失与输出](topics/06_DETECTION_TRAINING_ZH.md)
- [原理07：验证与能力边界](topics/07_VALIDATION_ZH.md)

以下保留训练入门与代码讲解：

- [专题一：深度学习训练基础](TRAINING_BASICS_ZH.md)
- [专题二：配置与数据流](CONFIG_AND_DATA_ZH.md)
- [专题三：train_vod.py 训练入口详解](../code_walkthrough/TRAIN_VOD_EXPLAINED_ZH.md)
- [专题四：R4Det 单帧模型流程详解](../code_walkthrough/R4Det_ONE_FRAME_EXPLAINED_ZH.md)
- [专题五：权重、训练与推理](CHECKPOINT_AND_INFERENCE_ZH.md)
- [R4Det 单帧代码学习、阶段状态与中间特征图解](../code_walkthrough/README_zh.md)
- [物理点云与点云融合学习路线](POINT_CLOUD_FUSION_LEARNING_ZH.md)
- [本机运行与复现手册](R4DET_RUNBOOK.md)
- [项目README](../README.md)
- [工作区总路线](../../docs/README.md)

[返回目录](#top)

<a id="order"></a>

## 4. 推荐执行顺序

1. 复跑 FP32/FP16 smoke；
2. 阅读配置和 `extract_pts_feat/extract_img_feat`；
3. 跟踪 radar depth 和 image-to-BEV；
4. 分别理解 PDF、DGTF、IGDR；
5. 用小数据过拟合训练链；
6. 获取真实同步数据后复现正式验证；
7. 建立 RadarOnly、CameraOnly、Fusion；
8. 接入 Radar_Process 统一点合同；
9. 做点丢失、相机失效和标定扰动；
10. 输出统一 Detection/RadarMeasurement 给跟踪工程。

[返回目录](#top)

<a id="done"></a>

## 5. 完成标准

- 每个点可回溯到物理候选和标定；
- RadarOnly/CameraOnly/Fusion均有真实同步数据指标；
- PDF、DGTF、IGDR逐项消融；
- Doppler、SNR和协方差特征有消融；
- 相机和点云失效时可安全降级；
- AP、Recall、定位RMSE、延迟按距离和点数切片报告。

[返回目录](#top) · [返回工作区总路线](../../docs/README.md)
