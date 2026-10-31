# 03 LSS、painting、跨模态融合与 LACA：两种坐标空间如何交换信息

[主要概览](../LEARNING_OVERVIEW_ZH.md) · [上一篇：图像与深度](02_IMAGE_DEPTH_ZH.md) · [下一篇：时序](04_TEMPORAL_DGTF_ZH.md)

前置知识是针孔投影和深度分布。本篇解释图像视角（PV）与鸟瞰视角（BEV）之间的交换，并区分几个经常被统称为“融合”的实际计算步骤。

## 1. LSS：从像素向三维空间分发特征

设像素 context 为 $\mathbf f(u,v)$，深度概率为 $p_k(u,v)$。Lift 构造

$$\mathbf V(u,v,k)=p_k(u,v)\mathbf f(u,v).$$

这是用同一语义向量的不同权重表示沿射线的不同位置。随后由内参反投影、外参和增强变换得到三维坐标，再把落入同一格的贡献聚合：

$$\mathbf F_{BEV}(i,j)=\sum_{(u,v,k)\mapsto(i,j)}\mathbf V(u,v,k).$$

该式说明聚合思想；实际轴与高度处理见 [ViewTransformerLSS](../../mmdet3d/models/img2bev/forward_projection/ViewTransformerLSS.py) 的 `get_geometry/voxel_pooling/forward`。实现先撤销图像增强，再经相机到雷达坐标变换及 BEV 增强。缺少其中一项，结果可能有正确 shape 却处在错误位置。

当前原始输出 `[1,256,216,248,1]`，主模型对高度轴求均值并交换平面轴，得到 `[1,256,248,216]`。高度轴当前为 1，不能据此假设所有配置都只有一层。LSS 将多个来源聚合到同一格，会丢失来源身份；深度错误还会把正确语义写入错误距离。

## 2. Painting：在给定三维参考位置读取视觉证据

当前 `use_sa_radarnet=True` 时，[extract_feat](../../mmdet3d/models/detectors/R4Det.py) 还调用 `voxelpainting_depth_aware()`。其输入参考位置来自 `voxelpainting_points` 网格，不应理解成只给原始测得雷达点上色。

```text
三维网格参考位置 → 投影到图像特征图
  → 按图像特征与深度概率读取证据
  → 整理高度和通道 → adaptive_collapse_conv
  → paint_bev 与原始 radar_bev 拼接 → PaintBEVFusion
```

所以跨模态融合模块收到的 radar 分支特征，在当前配置下已经经过视觉 painting。若要构造 RadarOnly，不能只关闭最后一次融合：还必须审计这条更早进入的图像路径。网格视觉特征也不等于雷达新测得了一个点。

## 3. Cross_Modal_Fusion：互相提供空间权重

[Cross_Modal_Fusion](../../mmdet3d/models/necks/BEVCross_modal_attention.py) 沿通道分别取均值和最大值，拼接后用卷积与 sigmoid 生成空间权重。记图像、雷达特征为 $I,R$：

$$A_I=\sigma(Conv([mean_c(I),max_c(I)])),\quad
A_R=\sigma(Conv([mean_c(R),max_c(R)])).$$

然后交叉调制并降维：

$$F=ConvBNReLU([I\odot A_R,\ R\odot A_I]).$$

这是本地实现的具体公式；虽然属性名叫 `cross_attention`，该模块不是全局 Q/K/V 点积 attention。图像 256 通道与雷达 384 通道拼接后，压到 256 通道。权重是学习出的特征调制，不是经过标定的传感器可信度。

融合要求两路的米制范围、原点、网格步长和轴顺序一致。通道数可以投影，几何错位不能靠拼接消除。当前主模型在后续阶段还会交换 x/y 轴，应按每个调用边界追踪 shape。

## 4. LACA：以 BEV 为起点重新查找图像证据

LSS 的方向是“每个像素能贡献哪些 BEV 位置”；LACA 后向投影是“这个 BEV 位置应到哪里读取图像证据”。当前路径由 [BackwardProjection](../../mmdet3d/models/img2bev/backward_projection/backward_projection.py) 和其 transformer 模块实现，接收 BEV、图像特征、几何参数和深度分布。

可将几何引导采样的思想写成

$$\Delta\mathbf q_i\sim\sum_m a_{im}\,w_d(i,m)\,\mathbf f(\pi(\mathbf X_i)+\delta_{im}),$$

其中 $\pi$ 是几何投影，$\delta$ 是学习偏移，$a$ 是注意力权重，$w_d$ 表示深度约束。**这是概念示意，不是逐行复写实现**；实际多头、多层、残差与归一化应继续读 [deformable_cross_attention.py](../../mmdet3d/models/img2bev/backward_projection/transformer_utils/deformable_cross_attention.py)。

参考点缩小搜索范围，学习偏移允许局部纠正，但不保证恢复任意标定误差。出视野、遮挡或错深度仍会使采样无效。当前配置包含 3 层 cross、2 层 self；主调用将 `proposal` 设置为全一布尔 BEV mask，不能仅凭 FRPN 名字认定此处只处理前景格。

## 5. 四步在主链中的位置

```text
图像 → 深度分布 → LSS Camera BEV ──────────────────┐
图像/深度 → 网格 painting → 与原始 Radar BEV 融合 ─┴→ Cross_Modal_Fusion
                                                    → 条件 DGTF → LACA
```

论文层面的 PDF 名称不能机械等同于一个 `Cross_Modal_Fusion` 类。这里按已核对的本地计算路径解释各步骤。

自检：解释为什么 painting 后的雷达特征不再是纯 RadarOnly；对照 [trace](../../code_walkthrough/figures/r4det_trace.json)，找到 LSS 的五维输出、跨模态输出和 LACA 的 `[1,53568,256]`，核算 $53568=216\times248$。随后读[时序专题](04_TEMPORAL_DGTF_ZH.md)。
