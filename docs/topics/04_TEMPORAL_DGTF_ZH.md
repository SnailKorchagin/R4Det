# 04 DGTF 时序融合：历史信息怎样对齐、筛选并进入当前帧

[主要概览](../LEARNING_OVERVIEW_ZH.md) · [上一篇：投影与融合](03_PROJECTION_FUSION_ZH.md) · [下一篇：实例细化](05_INSTANCE_IGDR_ZH.md)

本篇假定已理解 BEV 特征图。时序模块位于跨模态融合之后、LACA 之前，试图借历史证据补充当前稀疏观测。关键问题是历史目标位置已变，旧特征不能原样叠加。

## 1. 不对齐为什么会拖影

若目标与自车的相对横向速度为 4 m/s，两帧间隔 0.1 s，位移约 0.4 m，在 0.32 m BEV 中相当于 1.25 格。直接相加会让同一目标同时在新旧位置出现响应。真实位移还包含自车旋转与透视关系，不能对整幅 BEV 统一平移一个固定格数解决所有运动。

当前 [TemporalDeformableFusion](../../mmdet3d/models/fusion_layers/temporal_r4det_fusion.py) 从当前/历史特征预测局部采样偏移，使用调制可变形卷积对齐，再通过门控合成结果。

## 2. 可变形卷积改变什么

普通卷积在固定邻域位置 $p+p_k$ 采样；可变形卷积额外预测 $\Delta p_k$ 与幅度 mask $m_k$：

$$H_a(p)=\sum_k W_k\,H_{prev}(p+p_k+\Delta p_k)\,m_k.$$

非整数位置通过插值获得特征。当前位置与历史位置拼接后输入 `offset_mask_generator`；当前 $3\times3$ 卷积、1 个 deform group 共输出 $2\times9+9=27$ 通道，其中 18 个 offset 通道和 9 个 sigmoid mask 通道。

mask 描述采样贡献，offset 描述采样位置，它们不是同一个“注意力”。初始化时 offset/mask 生成卷积的参数全零，因此初始 offset 为 0、mask 为 0.5；后续卷积权重仍参与运算，所以这不等价于初始直接复制历史。

## 3. 门控递归形式怎样取舍新旧信息

令当前特征为 $C$，对齐历史为 $H_a$，源码计算为

$$\begin{aligned}
z&=\sigma(Conv_z([C,H_a])),\\
r&=\sigma(Conv_r([C,H_a])),\\
\tilde H&=Conv_h([C,r\odot H_a]),\\
H&=(1-z)\odot C+z\odot\tilde H,\\
F_{out}&=OutputConv(H).
\end{aligned}$$

这里 `Conv_h` 包含配置的归一化和 ReLU，并非必须采用教科书 GRU 的 tanh。$r$ 控制多少历史进入候选，$z$ 控制当前特征与候选的比例。当 $z\approx0$ 时，最终输出仍经过 `output_layer`，不能写成严格恒等于 $C$。

学习门控可抑制不相容历史，但不保证每个噪声或遮挡都被识别。所谓不显式依赖 pose 的特征对齐，不意味着不需要正确序列、时间间隔或运动分布。

## 4. 配置存在、模块调用与结果采用是三件事

[TJ4DDataset](../../mmdet3d/datasets/TJ4D_dataset.py) 准备前帧与当前帧信息；[simple_test](../../mmdet3d/models/detectors/R4Det.py) 在入口将 `self.prev_bev_feats` 设为 `None`，从本次输入中的前帧计算历史特征。它不是一个已经维护任意长历史的在线缓存接口。

`is_prev_frame_valid` 决定历史有效性。批内有有效历史时可以触发时序计算，`extract_feat` 再用 `torch.where` 让无效样本保留融合前的当前特征。因此“hook 被调用”与“每个样本采用时序结果”也需要区分。

已有 [STATUS](../../code_walkthrough/STATUS.md) 中 `temporal_fusion=SKIP`、调用次数为 0。该单帧证明其余链路可以执行，没有验证 DGTF 的数值行为或收益。

## 5. 怎样验证这个原理

先准备同序列真实相邻帧，记录时间差、有效标志和两帧 BEV；再检查模块调用、offset/mask 数值与输出差异。无效前帧样本应沿对应旁路保留当前特征。复制当前帧可检验接口，却不能检验运动对齐。

在统一数据与权重条件下比较时序开关，可以观察模型对历史的敏感性；要证明时序提高 AP，还要使用正式序列评估和匹配的训练/消融配置。此处说明验证方法，本次文档整理没有执行时序实验。

读完应能解释 27 个通道、两个 gate 和历史有效性三件事，再进入[实例引导细化](05_INSTANCE_IGDR_ZH.md)。
