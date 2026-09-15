# R4Det 本机运行记录

## 结论

R4Det 已在本机 RTX 4060 Ti 上完成两次单帧端到端推理：默认 FP32 和 MMCV FP16 都输出 `R4DET_SMOKE_OK`。验证链路包括 TJ4D 数据读取、图像增强、雷达投影、雷达/视觉特征提取、DFA3D、BEV 融合、检测头和 3D NMS。

当前公开 smoke 使用真实 TJ4D 雷达帧 `070070`、官方标定、官方标签和官方 R4Det 权重。TJ4DRadSet 的同步相机原图受 NDA 约束，公开 Sample 不含相机，所以视觉输入是由该帧雷达按标定投影生成的代理图。这可以验证代码完整性，不能用于复现论文 AP。

## 已验证环境

| 项目 | 版本 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Ti 8GB |
| Python | 3.8.20 |
| PyTorch | 2.0.1+cu118 |
| TorchVision | 0.15.2+cu118 |
| CUDA Toolkit | 11.8 |
| MMCV | 1.7.2 |
| MMDetection | 2.28.2 |
| R4Det 参数量 | 103,076,746 |

CUDA 扩展已针对 `sm_89` 编译，包括 R4Det/MMDetection3D ops、DFA3D、MultiScaleDeformableAttention 和 localattention。

## 一键复跑

在仓库根目录执行：

```bash
./run_r4det_smoke.sh
```

低显存 FP16 路径：

```bash
./run_r4det_smoke.sh --fp16
```

脚本先运行 `tools/prepare_r4det_public_smoke.py`，再运行 `tools/r4det_smoke_test.py`。成功标志是终端出现：

```text
PIPELINE_OK ...
R4DET_SMOKE_OK
```

逐阶段抓取真实中间 Tensor、生成三张学习图和状态表：

```bash
./run_r4det_walkthrough.sh
```

详细学习顺序和检验标准见：[R4Det 单帧代码学习与检验流程](../code_walkthrough/README_zh.md)。

## 本次实测结果

| 模式 | 单帧前向 | 峰值显存 | 候选框 |
| --- | ---: | ---: | ---: |
| FP32 | 0.846 秒 | 2298.4 MiB | 300 |
| FP16 | 0.899 秒 | 1921.1 MiB | 300 |

时间只包含已经完成模型构建和权重加载后的单帧前向，不能当作完整吞吐率基准。代理视觉输入与训练域不同，因此最高置信度很低，这是预期现象。

生成文件：

- `.smoke/r4det_result.json`：检测框、标签、分数、耗时和显存。
- `.smoke/r4det_bev.png`：灰色真实雷达点、红色 GT 框、绿色预测框。
- `data/TJ4D/public_smoke_manifest.json`：数据来源和“非原始相机”声明。
- `data/TJ4D/training/image_2/070070.png`：雷达投影代理图。

## 权重与样例校验

```text
final_tj4d.pth   b3950150903c007a2416efde591ea53227bae2238da3e2ec51aec88a7b0e699f
train info pkl   0eb60b281141afc22879efe008d23e78c485837e69cd912a126af7b752cccb11
070070 radar     55955028a97c535e489378edf768c6f722969b7aa1a722cdf7adc12bee073c00
```

官方 checkpoint 含 1418 个张量，已与修正后的模型严格加载，0 个 missing key、0 个 unexpected key。

## 为跑通所做的兼容修复

1. 作者 checkpoint 使用 `ifgd_fusion`，源码却调用 `igdr_fusion`；统一为 checkpoint 和 optimizer 配置采用的 `ifgd_fusion`。
2. 将仓库中实际未随代码提供的 Detectron2、AV2、VoD 和 FlashAttention 可选依赖改为按需导入，避免 TJ4D-R4Det 被无关模块阻塞。
3. 修复 FP16 中雷达 painting 和 backward projection 新建张量的 dtype 不一致。
4. DFA3D 的 depth-score CUDA 核没有 Half dispatch；仅把该算子局部提升为 FP32，输出再转回 FP16。
5. 删除 `CreateDepthFromRaDAR` 每帧写出 `lidar2img.npy` 和 `pts_hom.npy` 的调试副作用。

## 如何做正式论文复现

拿到 TJ4DRadSet 授权数据后，把完整数据按配置放入 `data/TJ4D/`，至少需要 `training/image_2`、`training/velodyne_reduced` 和对应的 info pkl/辅助文件。然后用官方 val info 运行 `tools/test_vod.py` 做全验证集评估。公开代理帧只能证明软件链路工作，不能代替真实相机完成 AP 对比。

上游资源：

- R4Det 代码：<https://github.com/VDIGPKU/R4Det>
- R4Det 权重和附加文件：<https://huggingface.co/Hoeyy/R4Det>
- TJ4DRadSet：<https://github.com/TJRadarLab/TJ4DRadSet>
