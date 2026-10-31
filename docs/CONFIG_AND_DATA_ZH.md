# 专题二：配置与数据流

[返回学习总览](LEARNING_OVERVIEW_ZH.md) · [上一专题：训练基础](TRAINING_BASICS_ZH.md) · [下一专题：训练入口](../code_walkthrough/TRAIN_VOD_EXPLAINED_ZH.md)

本专题回答：“配置文件写了 `data` 和 `model`，程序如何把这些字典变成真正的对象？”以 [TJ4D 训练配置](../configs/r4det/TJ4D-R4Det_det3d_2x4_12e.py) 为例。

## 1. 配置不是模型，也不是数据本身

```python
model = dict(type='R4Det', ...)
data = dict(train=dict(type='RepeatDataset', times=2,
                       dataset=dict(type='TJ4DDataset', ...)))
optimizer = dict(type='AdamW', lr=0.0002, ...)
runner = dict(type='EpochBasedRunner', max_epochs=18)
```

这些是 Python 配置对象。`type` 表示要查找的注册类名称。`build_model(cfg.model)` 和 `build_dataset(cfg.data.train)` 才会构造对象；训练框架再根据 `optimizer`、`runner` 等配置构造训练部件。该配置没有 `_base_`，当前实验设置主要在同一文件内。

## 2. 数据文件来自哪里

当前训练配置的关键值：

```text
data_root = data/TJ4D/
ann_file = data/TJ4D/TJ4D_infos_train.pkl
split = training
pts_prefix = velodyne_reduced
dataset type = TJ4DDataset
```

`.pkl` 提供样本索引和标注信息；数据集类再组合样本路径，取出图像、雷达等文件。`data_root` 是相对运行目录的路径。训练 pipeline 中 `LoadMyDepthFromFile` 还指定了独立的绝对路径，若换机器需要单独核对。

## 3. build_dataset() 一层层做了什么

[train_vod.py](../tools/train_vod.py) 调用：

```python
datasets = [build_dataset(cfg.data.train)]
```

[builder.py](../mmdet3d/datasets/builder.py) 看到外层 `type='RepeatDataset'`，就递归构建内部数据集，再用 `times=2` 包装：

```text
cfg.data.train
  → RepeatDataset 配置
      → TJ4DDataset 配置
          → TJ4DDataset 实例
      → RepeatDataset(TJ4DDataset, 2)
  → 放进 datasets 列表
```

`TJ4DDataset` 的定义在 [TJ4D_dataset.py](../mmdet3d/datasets/TJ4D_dataset.py)，继承 [Custom3DDataset](../mmdet3d/datasets/custom_3d.py)。初始化会读取数据索引和标注，并构造 pipeline。`RepeatDataset` 扩大逻辑数据集长度，不复制原始点云和图片。

## 4. Dataset、Pipeline、DataLoader 各做什么

| 部件 | 一个具体动作 | 对应位置 |
|---|---|---|
| Dataset | 给定索引，找到这一条样本及其标注 | `TJ4D_dataset.py` |
| Pipeline | 顺序加载、增强、归一化、打包样本 | 配置中的 `train_pipeline`，实现位于 `datasets/pipelines/` |
| DataLoader | 取样、启动 worker、组成 batch | 后续训练框架创建 |

**构建 Dataset 时还没有把全部原始图像和点云加载成 batch。** 真正训练取样时，DataLoader 调用数据集取样，流水线再处理样本。

TJ4D 的当前实现会判断相邻上一帧是否有效，并打包前一帧与当前帧。模型训练函数会拆开这两帧，按有效性决定是否融合前帧 BEV。详见 [TJ4D 数据集](../mmdet3d/datasets/TJ4D_dataset.py) 和 [R4Det.forward_train()](../mmdet3d/models/detectors/R4Det.py)。

## 5. Pipeline 从左到右读

当前训练 pipeline 可按用途分组：

```text
读取：LoadPointsFromFile / LoadImageFromFile / LoadAnnotations3D
额外监督：loadSegmentation / LoadVLSAMAnnotations / LoadMyDepthFromFile
增强：ImageAug3D2 / GlobalRotScaleTransFlipAll
过滤与整理：PointsRangeFilter / ObjectRangeFilter / PointShuffle / Normalize / Pad
深度与掩码：CreateDepthFromLiDAR / CreateDepthFromRaDAR / gen2DMask
交付模型：DefaultFormatBundle3D / CustomCollect3D2
```

每个 `dict(type='...')` 通过注册机制构造成一步处理。前一步改变样本字典，后一步读取更新后的字典。最后 `CustomCollect3D2` 决定哪些字段作为模型输入、哪些放在 `img_metas` 中。不要把 pipeline 的先后顺序当成随意的清单：几何增强后的矩阵和标注必须保持一致。

## 6. Model 的 `type` 如何找到源码

`build_model(cfg.model)` 根据 `type='R4Det'` 找到 [R4Det 类](../mmdet3d/models/detectors/R4Det.py)。这个类又按子配置创建 ResNet、FPN、深度网络、雷达 Pillar 网络、时序融合及检测头。读模型配置时先找最外层 `type`，再把子配置视为模型的组成模块，不必从第一天开始掌握每一个模块的内部公式。

## 7. 如何在 IDE 中追一条线

1. 在配置中找到 `data.train.type`，跳到 `build_dataset()`。
2. 看 `RepeatDataset` 分支怎样递归调用自己。
3. 找 `TJ4DDataset.__init__()`、样本准备/取样代码。
4. 回到配置，逐个看 `train_pipeline` 的 `type`；点进同名类的实现。
5. 找最终收集的 `keys` 和 `meta_keys`，再到 `R4Det.forward_train()` 看这些字段如何使用。

一次只追一个字段，例如 `points`：**配置声明 → 数据集提供路径 → 加载并过滤点 → 收集成 batch → 模型体素化**。能讲清一条线，再追 `img`、`radar_depth`、`gt_bboxes_3d`。
