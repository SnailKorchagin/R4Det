# R4Det：物理点云与点云融合

[R4Det 文档中心](README.md) · [工作区总路线](../../docs/README.md)

## 本页目录

- [1. 本章目的](#purpose)
- [2. 点云合同](#contract)
- [3. RadarOnly基线](#radar-only)
- [4. R4Det学习路径](#r4det)
- [5. 点云—相机融合实验](#experiments)
- [6. 数据与执行限制](#data)
- [7. 阶段门禁](#gate)

<a id="purpose"></a>

## 1. 本章目的

把 ADC/OMP/DML 候选变成带物理单位和协方差的点云，建立 RadarOnly 检测，再使用 R4Det 学习点云—相机的深度、BEV、时序和实例融合。

点云路线的优势：

- 数据量小、实时性较好；
- 距离和径向速度有明确物理意义；
- 容易与传统检测器、跟踪器和 Autoware 对接。

主要损失：

- 阈值前的低能量信息；
- 完整局部谱形；
- Doppler–elevation–azimuth 联合结构；
- 被 CFAR/峰值提取删除的弱目标。

因此点云链不能替代 Tensor 链，两者必须并行保留。

<a id="contract"></a>

## 2. 点云合同

第一版最少字段：

```text
x, y, z                 meter, radar/vehicle frame
radial_velocity         m/s, sign convention explicit
power                   calibrated or raw scale explicit
snr                     dB
noise_floor             dB
range/azimuth/elevation rad or meter
range/doppler bins      integer source indices
covariance              position/velocity uncertainty
timestamp
source                   CFAR-high/CFAR-low/OMP/DML/tensor
raw_roi_key
```

禁止仅输出 `[x,y,z,intensity]` 后丢失 Doppler、SNR、协方差和来源。

候选合并规则：

1. 先在原始 range/Doppler/angle 空间做 gating；
2. 同一物理峰的 high/low/OMP/DML 结果合并；
3. 物理估计主导 range、vr 和角度；
4. Tensor 更新 existence、artifact、class 和协方差；
5. 保存所有 source IDs，不能静默覆盖。

<a id="radar-only"></a>

## 3. RadarOnly 基线

学习顺序：

```text
P0  raw physical point visualization
P1  simple clustering + rule-based boxes
P2  PointPillars RadarOnly
P3  multi-frame point accumulation with ego-motion
P4  Doppler/SNR/covariance feature ablation
```

P2 输入消融：

```text
xyz
xyz + power
xyz + vr
xyz + vr + power + SNR
all + covariance encoding
high threshold only
high + verified low-threshold points
```

必须按距离和点数切片，防止整体 AP 掩盖远距/弱目标失败。

<a id="r4det"></a>

## 4. R4Det 学习路径

### 4.1 四个复现层级

```text
L0 环境：模型构建、CUDA算子、smoke forward
L1 权重：checkpoint严格加载、单帧推理
L2 训练：小数据过拟合、loss和反向
L3 指标：真实同步数据上的完整验证集
```

当前本机已达到 TJ4D 的 L1，但公开代理相机不能用于 L3。

### 4.2 源码阅读顺序

1. 配置中的输入、监督、点云范围和 voxel size；
2. `extract_pts_feat`：点→voxel/pillar→radar BEV；
3. `extract_img_feat`：图像 backbone/FPN；
4. radar-assisted depth 与 image-to-BEV；
5. PDF/cross-modal fusion；
6. DGTF temporal fusion；
7. IGDR instance refinement；
8. detection head 与 NMS；
9. `forward_train` 全部 loss；
10. `simple_test` 完整输出。

### 4.3 必须能回答

- 雷达点怎样帮助 camera depth？
- 相机 PV 特征怎样转换到 BEV？
- Radar BEV 与 Camera BEV 如何对齐？
- 历史 BEV 无 pose 时怎样对齐，什么时候会失败？
- gate 如何抑制错误历史信息？
- instance prototype 如何净化 BEV？
- 目标没有雷达点时，模型依赖什么？
- 点云过稀或 camera invalid 时如何降级？

<a id="experiments"></a>

## 5. 点云—相机融合实验

固定消融：

```text
PC0  RadarOnly PointPillars
PC1  CameraOnly BEV
PC2  simple concatenation
PC3  radar painting into camera/PV
PC4  Camera BEV + Radar BEV cross-modal fusion
PC5  PC4 + temporal fusion
PC6  PC5 + instance refinement
PC7  full R4Det
```

失效实验：

```text
camera black/blur/overexposure
camera timestamp offset
extrinsic perturbation
radar point dropout 25/50/75%
wrong Doppler sign
missing history frame
ego-motion perturbation
```

每组同时报告：

- AP/Recall；
- 近/中/远距离；
- 每目标雷达点数；
- day/night/weather；
- localization RMSE；
- 延迟和显存；
- camera invalid 时相对 RadarOnly 的退化。

<a id="data"></a>

## 6. 数据与执行限制

### 当前本机 smoke

```bash
cd /home/snailkorchagin/Radar/R4Det
./run_r4det_smoke.sh
./run_r4det_smoke.sh --fp16
```

成功只能说明软件链路、权重和单帧输入可运行。

### 正式数据顺序

1. RadarScenes：点级标签、track ID，先做点云与跟踪；
2. View of Delft：4D radar point cloud + camera + lidar + odometry，做公开融合；
3. TJ4DRadSet：与当前 R4Det 配置最接近，需取得完整同步相机；
4. 自采 ADC：用统一 point schema 接入，与公开数据只共享接口，不共享未经验证的归一化。

注意：当前本地 R4Det 上游 VoD 配置不完整，使用 VoD 前要先补齐配置、数据转换和权重，不得声称已有一键运行能力。

<a id="gate"></a>

## 7. 阶段门禁

- [ ] 每个点可回溯到 ADC/候选和标定版本；
- [ ] RadarOnly 在小数据上可以过拟合；
- [ ] 点特征消融解释 vr/SNR/covariance 的贡献；
- [ ] R4Det 正式实验使用原始同步相机而非代理图；
- [ ] RadarOnly/CameraOnly/Fusion 使用同一 split 和评价协议；
- [ ] camera invalid 时结果确定性退回 RadarOnly 或受控降级；
- [ ] 点云路线与 Tensor 路线的输入损失被明确记录。

[R4Det 文档中心](README.md) · [工作区总路线](../../docs/README.md)
