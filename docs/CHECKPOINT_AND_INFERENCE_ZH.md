# 专题五：权重、训练与推理

[返回学习总览](LEARNING_OVERVIEW_ZH.md) · [训练入口详解](../code_walkthrough/TRAIN_VOD_EXPLAINED_ZH.md) · [模型单帧流程](../code_walkthrough/R4Det_ONE_FRAME_EXPLAINED_ZH.md)

训练的目标是获得一组能完成检测的模型参数。参数存入 checkpoint 后，推理程序按配置重建同样的模型结构，加载这些数值，再处理输入。

## 1. 三份东西分别是什么

| 对象 | 保存什么 | 变化方式 |
|---|---|---|
| 配置 `.py` | 网络结构选择、数据路径、学习率、训练轮数等 | 人编辑或命令行覆盖 |
| 模型代码 `R4Det.py` | 前向计算、训练损失、推理逻辑 | 人修改源码 |
| checkpoint `.pth` | 模型参数；可能还含优化器状态、epoch、元信息 | 训练过程保存 |

假设模型有一个权重张量 `W`。源码规定如何用 `W` 计算输出；训练把 `W` 的数值从初始值更新到学到的值；`.pth` 保存更新后的数值。源码不会因为训练而自动改变。

## 2. 训练怎样产生 checkpoint

```text
tools/train_vod.py → build_model + build_dataset
  → mmdet3d/apis/train_vod.py → mmdet.apis.train_detector
  → 训练框架反复调用模型、计算 loss、反向传播、更新参数
  → checkpoint hook 按配置保存
```

当前配置 `checkpoint_config.interval=1`，表示按每个 epoch 的间隔保存。保存文件的具体命名和内容由实际训练框架及其 hook 决定。

## 3. load_from 与 resume_from

| 配置 | 通常含义 | 适合场景 |
|---|---|---|
| `load_from` | 加载模型权重作为本次训练的起点 | 预训练后继续训练新实验 |
| `resume_from` | 恢复中断的训练，包括可用的进度和优化器状态 | 上次训练中途停止 |

当前 TJ4D 配置有 `load_from='.../epoch_2.pth'`、`resume_from=None`。因此它意图从已有权重开始一次训练，而不是恢复那个 checkpoint 所记录的完整训练进度。两者最终行为以实际使用的训练框架实现和 checkpoint 内容为准。

脚本中 `load_radar_from`、`load_lidar_from` 是另一类加载：把单分支模型的参数名映射到 R4Det 对应分支；当前配置没有设置它们。

## 4. 推理入口怎样使用权重

[tools/test_vod.py](../tools/test_vod.py) 会先根据测试配置构建数据集与模型，再在普通 checkpoint 分支调用：

```python
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
```

`load_checkpoint()` 把 checkpoint 中的参数复制进模型对象。之后测试流程调用模型，进入 [R4Det.simple_test()](../mmdet3d/models/detectors/R4Det.py)。该方法依赖共享的 `extract_feat()`，并进行检测头预测与后处理。

匹配的配置和 checkpoint 很重要：类别数、层名、通道数等结构改变后，旧权重可能无法完整加载，或结果不符合预期。

## 5. “给一个新输入就能推理”在本项目里是什么意思

概念上，新输入通过与训练兼容的预处理，交给已加载权重的模型，得到检测结果。当前 R4Det 实现还要求特定输入合同：雷达点、图像、相机与雷达几何矩阵、前帧有效性等元信息。`simple_test()` 的当前代码会拆分前帧和当前帧，也访问若干标注相关字段。因此部署到完全无标注的新场景，需要先核对并适配接口，不能只传一张图片就调用。

当前公开单帧演示使用雷达投影生成的代理相机图；它证明代码链可以运行，不能证明真实相机融合的性能。实测边界见 [本机运行手册](R4DET_RUNBOOK.md)。

## 6. 用三个问题检验理解

1. 训练后，`R4Det.py` 文件里的代码会自动变化吗？不会；变化的是模型对象中的参数值，并被保存到 checkpoint。
2. 推理时是否必须重新训练？不用；构建相同结构并加载合适的 checkpoint。
3. 推理会继续更新权重吗？通常不会；模型在评估模式下进行前向计算，不做优化器更新。
