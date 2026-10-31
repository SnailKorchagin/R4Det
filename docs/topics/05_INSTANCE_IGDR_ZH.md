# 05 IGDR 实例引导细化：把“一个物体”的特征写回 BEV

[主要概览](../LEARNING_OVERVIEW_ZH.md) · [上一篇：时序](04_TEMPORAL_DGTF_ZH.md) · [下一篇：检测与监督](06_DETECTION_TRAINING_ZH.md)

本篇需要理解特征图、池化和 sigmoid/softmax。IGDR 位于共享特征提取之后、3D 检测头之前，使用图像实例及其 BEV 分布，对已有融合特征做条件调制。

## 1. 实例和语义有什么区别

语义回答“哪些像素属于汽车”，实例进一步区分“这是汽车 A，那是汽车 B”。当前图像支路的 RPN 生成候选区域，RoI 模块抽取每个区域的局部特征并预测框、类别和 mask。RoIAlign 把大小不同的区域整理成固定大小特征，便于后续统一处理。

保存的记录有 `[2,256,14,14]` 的实例特征，表示两个区域，每个区域 256 通道、14×14 空间位置；它不是两辆真实物体已被正确检出的证据。

## 2. 从实例特征到 BEV 特征场

本地 [IGDRModule](../../mmdet3d/models/detectors/R4Det.py) 的对象属性名是 `ifgd_fusion`。代码对每个实例 $n$ 做空间平均，得到 prototype：

$$\mathbf e_n=\frac1{hw}\sum_{u,v}\mathbf E_n(u,v).$$

prototype 是一个对象区域的学习特征摘要，不是目标 RCS、物理散射中心或独立测量。`_build_bev_instance_map()` 用实例 mask、位置和深度等信息构造 $S_{BEV}$，把图像实例与 BEV 位置联系起来。

IGDR 将 $S$ 的平面轴重排后，在实例维上执行温度为 $T=0.01$ 的 softmax：

$$A_n(i,j)=\frac{\exp(S_n(i,j)/T)}{\sum_m\exp(S_m(i,j)/T)},\qquad
\mathbf E_{BEV}(i,j)=\sum_n A_n(i,j)\mathbf e_n.$$

较低温度使分配尖锐，但若所有实例分数相同，则仍是均匀分配；若只有一个实例，其 softmax 恒为 1。这说明“没有明显实例支持”并不会仅靠该 softmax 自动产生零向量，后面的空间 gate 很重要。

## 3. gamma、beta 与 gate 各做什么

实例特征场通过两个 1×1 卷积生成逐通道调制参数：

$$\gamma=Conv_\gamma(E_{BEV}),\quad\beta=Conv_\beta(E_{BEV}),\quad
F_{fixed}=F\odot\gamma+\beta.$$

再由实例图生成空间 gate：

$$G=\sigma(Conv(\sum_n S_n)),\qquad F_{final}=(1-G)\odot F+G\odot F_{fixed}.$$

$\gamma$ 是乘性缩放，$\beta$ 是加性偏移，$G$ 决定采用多少调制后的结果。源码把 gamma 卷积权重初始化为 0、bias 为 1，把 beta 权重/bias 初始化为 0，所以初始 $F_{fixed}=F$，该调制初始保持输入；加载训练权重后不再保证这一性质。

当前配置实例通道和 BEV 通道都是 256，`project_layer` 为 Identity。分析其他通道配置前需重新检查实现，不能只按类名假设所有维数都可自动匹配。

## 4. 什么时候执行，为什么可能失效

`simple_test()` 只有在实例特征非空等条件满足时才调用 `ifgd_fusion`；训练路径还检查预测 mask 等条件。当前 [STATUS](../../code_walkthrough/STATUS.md) 显示该分支执行一次，输入两份实例特征。

如果图像实例漏检、mask 合并了相邻目标，或者深度把实例投到错误 BEV 位置，调制也可能强化错误区域。IGDR 的“细化/净化”指学习特征的调整，不能直接推导它消除了雷达相干旁瓣或多径。

## 5. 学习检查

手算两个实例 $S_1=S_2$ 时的 $A_1=A_2=0.5$，以及单实例时的分配；再把初始 gamma/beta 代入，验证 $F_{final}=F$。读代码时分别追踪实例特征、空间分配图和 gate，避免把三者都称作 mask。

需要验证实际效果时，检查有/无实例两条路径以及调制前后的特征差，再在真实同步数据上判断检测收益。接下来读[检测头与监督](06_DETECTION_TRAINING_ZH.md)。
