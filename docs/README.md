# R4Det 文档中心

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

该工程接收 Radar_Process 输出的：

```text
[x,y,z,radial_velocity,power,SNR,covariance,source]
```

[返回目录](#top)

<a id="status"></a>

## 2. 当前复现状态

已验证：

- RTX 4060 Ti上的FP32/FP16单帧推理；
- 作者TJ4D完整checkpoint严格加载；
- 真实TJ4D雷达、标定和标签；
- 雷达/视觉/深度/BEV/融合/检测/NMS完整链路。

边界：

- 当前公开 smoke 的视觉输入是雷达投影代理图；
- 结果只证明软件链可运行，不能复现论文 AP；
- 正式验证需要完整同步相机和官方 split；
- 本地上游 VoD 配置不完整，不能假定已具备一键 VoD 训练能力。

[返回目录](#top)

<a id="documents"></a>

## 3. 详细文档

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
