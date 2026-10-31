# 06 检测头、损失与输出：BEV 如何变成类别和 3D 框

[主要概览](../LEARNING_OVERVIEW_ZH.md) · [上一篇：实例细化](05_INSTANCE_IGDR_ZH.md) · [下一篇：验证与边界](07_VALIDATION_ZH.md)

本篇从最终 `[B,256,248,216]` BEV 出发，解释预测、监督和后处理。训练框架与 checkpoint 的入门知识分别见[训练基础](../TRAINING_BASICS_ZH.md)和[权重与推理](../CHECKPOINT_AND_INFERENCE_ZH.md)。

## 1. Anchor 是参照框

当前 [配置](../../configs/r4det/TJ4D-R4Det_det3d_2x4_12e.py) 选择 `Anchor3DHead`，而不是同文件中出现的 `DynamicAnchor3DHead`。类别是 Pedestrian、Cyclist、Car、Truck；四组尺寸、两个方向组成每位置 8 个 anchor。anchor 提供位置、尺寸和方向的参照，网络预测相对它的修正量。

因此 [状态记录](../../code_walkthrough/STATUS.md) 的三个输出分别是：

| 输出 | shape | 通道解释 |
|---|---|---|
| 分类 | `[1,32,248,216]` | 8 个 anchor × 4 类 |
| 框残差 | `[1,56,248,216]` | 8 × 7 个框编码量 |
| 方向分类 | `[1,16,248,216]` | 8 × 2 个方向 bin |

配置顶部的 `code_size` 变量不等于此头最终实际使用的框维度；应查看传入 bbox coder 的参数和输出。当前 coder 默认 7 维，不因为输入点含径向速度就自动输出目标完整速度。

## 2. 框残差为什么需要归一化

[DeltaXYZWLHRBBoxCoder](../../mmdet3d/core/bbox/coders/delta_xyzwhlr_bbox_coder.py) 用 anchor 的尺寸归一化位移，尺寸用对数比表达。令 $d_a=\sqrt{l_a^2+w_a^2}$：

$$t_x=(x_g-x_a)/d_a,\quad t_y=(y_g-y_a)/d_a,\quad
 t_w=\log(w_g/w_a),\quad t_l=\log(l_g/l_a).$$

解码时 $x_g=x_a+t_xd_a$，$w_g=w_a\exp(t_w)$。例如 $w_a=2,l_a=4$，$t_x=0.1$ 表示沿 x 修正约 0.447 m，而不是 0.1 m；$t_w=\log1.1$ 表示宽度增大 10%。

z 编码先把底面 z 转成中心 z，用 anchor 高度归一化；解码后再转回底面 z。忽略这一点会让画框与评测产生半个高度的偏移。yaw 残差、正弦差处理和方向分类共同处理角度周期，不能把 yaw 当普通无界位置值。

anchor 位置由自己的 `anchor_generator.ranges` 生成，当前范围与点云裁剪范围并非完全相同；不要从 BEV 索引直接假定 anchor 中心恰好等于 pillar 中心。

## 3. 训练时什么约束这些预测

[Anchor3DHead](../../mmdet3d/models/dense_heads/anchor3d_head.py) 按分配规则把 anchor 与 GT 匹配，得到正样本、负样本及忽略项，再计算分类、回归和方向损失。并非所有格都有一个真实目标，也不是每个输出框都直接拿来和所有 GT 比较。

当前分类使用 FocalLoss。其二分类形式可写为

$$L_{focal}=-\alpha_t(1-p_t)^\gamma\log p_t.$$

容易分类的样本 $p_t\approx1$ 时权重变小，减少海量容易负样本的主导作用；这不等于自动消除类别不平衡。当前配置 $\gamma=2,\alpha=0.25$。

框回归使用 Smooth L1，误差 $e$ 的核心形式为

$$L(e)=\begin{cases}e^2/(2\beta),&|e|<\beta\\|e|-\beta/2,&\text{其他}\end{cases}.$$

当前 $\beta=1/9$、框损失权重 2，方向交叉熵权重 0.2。小误差区平滑，大误差区线性增长。模型还可能计算深度、图像框/mask、BEV 前景等辅助损失，具体由 `forward_train()` 中分支和返回的 loss 字典决定。

总损失可概括为 $L=\sum_i\lambda_iL_i$。辅助项帮助中间表示学习，但权重过大也可能让优化偏离最终检测目标。loss 下降只是优化现象，AP 必须独立评估。

## 4. decode 与 NMS 为什么还需要

推理先把残差解码成实际框，再按分数和 NMS 筛选。NMS 通常按分数保留高分框并压制重叠重复框；邻近真目标也可能被误抑制，因此改变阈值会改变召回与误报的平衡。

当前记录有 300 个候选，反映候选生成及后处理配置，不代表有 300 个真实物体。分类得分来自学习目标，不是经过校准的存在概率；同样不能把一张特征热力图上的峰值当成目标置信度。

## 5. 训练和推理如何连接

配置选结构与超参数，源码定义运算；运行时模型对象持有可学习参数；优化器依据 loss 更新参数；checkpoint 保存参数及可能的训练状态。`load_from` 提供权重起点，`resume_from` 用于恢复可用训练进度，两者不能混用概念。

本地 `simple_test()` 还拆分并写入 GT 相关元数据。正式部署前要逐字段区分只用于显示/评价的标签与进入预测的依赖，做标签删除或扰动检查；不能仅凭函数叫 test 就断言完全无标签依赖。

自检：从 8 个 anchor 推出 32/56/16 通道；解码一个位移残差；解释为什么 loss、候选数和 AP 是三个不同量。随后进入[验证专题](07_VALIDATION_ZH.md)。
