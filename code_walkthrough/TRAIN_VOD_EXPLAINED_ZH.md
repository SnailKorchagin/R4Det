# train_vod.py 训练入口详解

[返回学习文档](README_zh.md)

本文对应当前仓库的 [tools/train_vod.py](../tools/train_vod.py)，结合 [TJ4D 配置](../configs/r4det/TJ4D-R4Det_det3d_2x4_12e.py) 解释。行号以撰写时版本为准。

## 1. 先确认：谁训练，谁推理，谁加载权重？

你的理解大方向正确：先训练得到权重，再用这些权重处理新输入。但需要补充：**R4Det.py 同时定义训练和推理所用的模型，并不是只负责推理；权重通常由外部入口加载到模型对象中。**

| 文件或对象 | 职责 |
|---|---|
| `tools/train_vod.py` | 启动和组织训练：读取配置、构造模型和数据集、调用训练框架 |
| `mmdet3d/models/detectors/R4Det.py` | 定义模型结构、特征提取、训练损失与推理流程 |
| `mmdet3d/apis/train_vod.py` | 根据模型类型转交给训练函数；R4Det 走 `mmdet.apis.train_detector` |
| 训练框架的优化器和 Runner | 反向传播、更新参数、调度学习率、保存 checkpoint |
| `tools/test_vod.py` | 构造测试模型、加载指定 checkpoint、运行测试和评估 |
| `.pth` checkpoint | 保存训练得到的参数，以及可能存在的优化器状态、训练进度、元信息 |

训练过程：

```text
配置 + 带标注训练数据
    → train_vod.py 构造 R4Det 模型
    → R4Det.forward_train() 计算预测和 loss
    → 训练框架反向传播，优化器更新可训练参数
    → checkpoint hook 保存 epoch_x.pth
```

推理过程：

```text
配置 + epoch_x.pth + 新输入
    → 测试/推理入口构造同样的 R4Det 模型
    → load_checkpoint(model, checkpoint_path) 将参数加载到模型对象
    → 模型进入评估模式，通常关闭梯度
    → R4Det.simple_test() 等推理路径
    → 输出类别、置信度和 3D 检测框
```

可以把配置看成“结构和运行设置”，把 `R4Det.py` 看成“模型的实现”，把 checkpoint 看成“模型学到的数值”。训练不会把学到的权重写回 `R4Det.py` 源码。

“参数”和“权重”在这里通常不是两份产物：权重是模型参数的一部分，参数还包括 bias 等。学习率、batch size、epoch 数等属于超参数，并非反向传播学习得到。

### 当前项目的新输入还有哪些要求？

当前 `simple_test()` 实现会拆分前一帧与当前帧，读取几何元信息，并访问传入的部分标注字段。因此“有新输入就推理”在概念上成立，但当前仓库接口不是任意传一张图片即可：需要按数据管线准备雷达、图像、标定、时序字段，以及当前实现需要的其他字段。部署到完全无标注的新数据时，应检查并适配这些依赖。

## 2. 整体执行顺序

```text
执行 tools/train_vod.py
  → 顶部初始化与导入
  → main()
      → parse_args()
      → Config.fromfile()
      → 根据配置文件名复制评估源码
      → 合并命令行覆盖配置
      → 设置输出目录、meta_info、GPU
      → 初始化分布式环境
      → 保存配置、记录日志、设置种子
      → build_model()
      → build_dataset()
      → 准备 checkpoint 元信息
      → 可选加载雷达/LiDAR分支预训练权重
      → train_model_vod()
```

## 3. 第 1～26 行：初始化与导入

`training_old_version.use_old_version()` 设置项目自定义的旧版本标志；紧接着的断言检查该标志。之后又从 `mmdet3d` 导入同名模块，因此需要区分两个模块命名空间，不能假设它们共享全局变量。

`warnings.filterwarnings('ignore')` 屏蔽 warning；`torch.autograd.set_detect_anomaly(True)` 开启自动求导异常检测，方便定位反向传播异常，但会增加运行开销。

关键函数：

```python
from mmdet3d.apis import train_model_vod
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
```

分别负责训练调度、数据集构建和模型构建。顶部导入 `wandb`，但后面的初始化调用被注释；当前仍需要能导入该依赖。

## 4. 第 27～91 行：parse_args()

该函数读取命令行参数，返回 `args`。

| 参数 | 作用 |
|---|---|
| `--config` | 配置文件路径；默认值是另一个项目配置，运行 TJ4D 时应显式指定 |
| `--work-dir` | 输出日志和模型的目录 |
| `--resume-from` | 恢复训练的 checkpoint |
| `--no-validate` | 关闭训练期间评估 |
| `--gpus` / `--gpu-ids` | 非分布式 GPU 设置，两者互斥 |
| `--seed` | 随机种子，默认 0 |
| `--deterministic` | 请求相关后端使用确定性设置 |
| `--cfg-options` | 覆盖配置字段 |
| `--launcher` | `none`、`pytorch`、`slurm`、`mpi` |
| `--local-rank` / `--local_rank` | 当前进程在本机的编号 |
| `--autoscale-lr` | 按 GPU 数缩放学习率 |

`--options` 是旧参数，会转成 `--cfg-options`，二者不能同时使用。

例如，从 `R4Det/` 目录运行：

```bash
python tools/train_vod.py \
  --config configs/r4det/TJ4D-R4Det_det3d_2x4_12e.py \
  --work-dir work_dirs/my_experiment \
  --cfg-options data.samples_per_gpu=2 runner.max_epochs=6
```

这个例子会真正启动训练。若配置中有 `load_from`，仍会使用那个 checkpoint。

这里直接覆盖 `runner.max_epochs`：Python 配置读完后，顶层 `max_epochs` 与 `runner.max_epochs` 已经是独立值，修改前者不会重新执行赋值来同步后者。

脚本还在环境变量不存在时设置 `LOCAL_RANK`，供分布式初始化使用。

## 5. 第 126～154 行：读配置、切换评估源码

```python
cfg = Config.fromfile(args.config)
```

执行 Python 配置并生成配置对象，例如 `cfg.model.type == 'R4Det'`。此时还没有创建模型。

`project` 从配置文件名提取。当前文件得到 `TJ4D-R4Det_det3d_2x4_12e`。脚本按名字开头判断数据集，并复制文件：

```text
tools/eval_tools/TJ4D-eval.py
  → mmdet3d/core/evaluation/kitti_utils/eval.py
tools/eval_tools/TJ4D-kitti_dataset.py
  → mmdet3d/datasets/kitti_dataset.py
```

这是覆盖源码的磁盘操作。路径相对启动工作目录；而且 `mmdet3d` 已在顶部导入，复制不会自动重新加载当前进程已导入的模块。因此不能仅凭复制动作就断言所有已加载类都切换成功。

随后 `cfg.merge_from_dict(args.cfg_options)` 合并命令行配置；`import_modules_from_strings()` 处理 `custom_imports`，使配置中的类能够通过注册器被查找。

## 6. 第 155～190 行：目录、附加信息和分布式

`work_dir` 优先级：命令行 > 配置 > `./work_dirs/配置文件名`。

对当前配置，脚本补充：

```python
cfg.model.meta_info = dict(
    figures_path='输出目录/figures_path',
    project_name='TJ4D-R4Det_det3d_2x4_12e'
)
```

`R4Det.__init__()` 会使用它们设置图片输出路径、判断数据集类型。直接脱离入口构建模型时，也要考虑这个依赖。

`--resume-from` 会覆盖配置中的 `resume_from`。未指定 GPU 参数时，先设置单 GPU；分布式模式下调用 `init_dist()`，再按 `world_size` 更新 `cfg.gpu_ids`。`rank` 是全局进程编号。

自动学习率缩放公式：

```python
lr = lr * len(cfg.gpu_ids) / 8
```

该计算发生在分布式初始化之前，因此用到的 GPU 数未必是最终 `world_size`。

## 7. 第 191～231 行：日志、配置快照、随机种子

主要操作：

- 创建输出目录，使用 `cfg.dump()` 保存当前配置快照。
- 以时间戳命名日志文件。
- `collect_env()` 收集软件和运行环境信息。
- 记录配置、分布式状态、随机种子和实验名称。
- 调用 `set_random_seed()`。

这里的 `meta` 用于训练记录；`cfg.model.meta_info` 用于模型内部，两者职责不同。

脚本前面同时设置了 cuDNN 的 `deterministic=True` 和 `benchmark=True`，后面还会调用随机种子函数。不能只看其中一行就认定运行完全可复现。

## 8. 第 233～238 行：build_model()

```python
model = build_model(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))
model.init_weights()
```

`build_model()` 根据 `type='R4Det'` 找到注册类，执行 `R4Det.__init__()`，继续构建图像、雷达、融合和检测分支。

你的训练与测试设置位于 `cfg.model.train_cfg/test_cfg`；顶层 `cfg.get('train_cfg')` 为 `None` 不代表没有训练设置。

当前 `R4Det` 构造函数内部也调用了 `init_weights()`，所以代码层面存在两次调用；是否重复初始化要结合模块的初始化状态判断。

此处初始化参数不等于加载配置中的整模型 `load_from`。

## 9. 第 239 行：重点理解 build_dataset()

```python
datasets = [build_dataset(cfg.data.train)]
```

等价于：

```python
train_dataset = build_dataset(cfg.data.train)
datasets = [train_dataset]
```

外面的方括号是 Python 列表，用于后续 workflow；列表中的元素才是 Dataset。

### 9.1 输入配置

```python
train = dict(
    type='RepeatDataset',
    times=2,
    dataset=dict(
        type='TJ4DDataset',
        data_root='data/TJ4D/',
        ann_file='data/TJ4D/TJ4D_infos_train.pkl',
        pipeline=train_pipeline,
        # 其他字段省略
    )
)
```

### 9.2 递归构建

实现位于 [datasets/builder.py](../mmdet3d/datasets/builder.py)，核心分支：

```python
elif cfg['type'] == 'RepeatDataset':
    dataset = RepeatDataset(
        build_dataset(cfg['dataset'], default_args), cfg['times'])
```

所以结果是：

```text
build_dataset(RepeatDataset 配置)
  → build_dataset(TJ4DDataset 配置)
      → 注册器找到 TJ4DDataset
      → TJ4DDataset(...)
  → RepeatDataset(内部数据集, times=2)
```

数据集类位于 [TJ4D_dataset.py](../mmdet3d/datasets/TJ4D_dataset.py)，父类位于 [custom_3d.py](../mmdet3d/datasets/custom_3d.py)。初始化时读取索引、标注信息，并通过 `Compose(pipeline)` 创建处理流水线。

### 9.3 创建数据集不等于读取所有原始数据

```text
构建 Dataset：读索引和标注，准备 pipeline
训练时取 dataset[i]：按路径读取图片、点云等，执行 pipeline
TJ4D 当前实现：检查前帧有效性，打包前帧与当前帧
```

`RepeatDataset(times=2)` 将逻辑长度从 N 扩展到 2N，通过索引映射重复访问，不复制原始文件。随机增强可能使两次访问结果不同。

### 9.4 Dataset 与 DataLoader

| 对象 | 负责什么 |
|---|---|
| Dataset | 给定索引，返回处理后的样本 |
| DataLoader | 采样、多进程读取、拼 batch |
| `build_dataset()` | 创建 Dataset |
| 后续训练函数 | 构建并迭代 DataLoader |

因此第 239 行还没有形成训练 batch。当前配置的 `samples_per_gpu=4`、`workers_per_gpu=6` 在后续数据加载阶段使用。

## 10. 第 240～264 行：workflow 与 checkpoint 元信息

当 `len(cfg.workflow) == 2` 时，脚本深拷贝验证配置，替换为训练 pipeline，设 `test_mode=False`，并构建第二个数据集。

当前 `workflow=[('train', 1)]`，所以不进入这个分支。训练期间按周期计算检测指标由后续评估 hook 负责，不要求 workflow 有第二个阶段。

`cfg.checkpoint_config.meta` 保存软件版本、配置文本、类别等；`model.CLASSES = datasets[0].CLASSES` 为可视化等提供名称，不会因此重新构建分类层。

## 11. 第 98～124、265～284 行：分支预训练权重

`load_pretrained_model()` 读取 checkpoint，兼容其中的 `state_dict`、`model` 或直接参数字典，处理 `module.` 前缀，并按映射重命名参数。

雷达分支映射：

```text
voxel_encoder  → pts_voxel_encoder
middle_encoder → pts_middle_encoder
backbone       → pts_backbone
neck           → pts_neck
```

用于把单独雷达模型的权重加载到多模态模型的雷达分支。

`strict=False` 允许缺失或多余的键，但不代表同名参数形状不一致也能加载。

当前配置没有 `load_radar_from`、`load_lidar_from`，因此跳过这两段。配置中的 `load_from` 属于整模型 checkpoint，交给后续训练入口处理。

第 277 行 LiDAR 分支打印误用了 `cfg.load_radar_from`，仅设置 `load_lidar_from` 时可能在这里报错。

## 12. 第 287～294 行：交给训练框架

```python
train_model_vod(
    model, datasets, cfg,
    distributed=distributed,
    validate=(not args.no_validate),
    timestamp=timestamp,
    meta=meta)
```

真实调用链：

```text
tools/train_vod.py
  → mmdet3d/apis/train_vod.py::train_model_vod()
  → mmdet.apis.train_detector()
```

这里不是直接进入本仓库 `mmdet3d/apis/train.py`。DataLoader、优化器、Runner、hook、恢复与加载 checkpoint 的具体行为，应继续查看运行环境实际导入的 MMDetection 实现。

训练循环的职责是：取 batch → 前向计算 loss → 反向传播 → 更新参数 → 记录日志、评估、保存权重。R4Det 模型的训练逻辑位于 `R4Det.forward_train()`，特征主干位于 `extract_feat()`。

`load_from` 通常用于加载模型权重作为起点；`resume_from` 用于恢复训练进度及相关状态，两者不能混为一谈。

## 13. 第 295～305 行：异常与入口

`try` 从第 232 行开始。捕获异常后，脚本打印调用栈，并写入当前工作目录的 `rank_<编号>_error.log`，最后重新抛出。

顶部导入、读配置、复制源码等发生在 `try` 之前，不会由这段逻辑记录错误。

`if __name__ == '__main__': main()` 表示直接执行这个文件时启动主流程。

## 14. 对照当前配置记住这些值

| 设置 | 当前值或行为 |
|---|---|
| 模型 | R4Det，同时实现训练和推理 |
| 训练数据集 | RepeatDataset(TJ4DDataset)，重复 2 次 |
| 训练时长 | `runner.max_epochs=18`，不是文件名里的 12 |
| 每 GPU batch | 4 |
| 整模型初始权重 | `load_from=work_dirs/TJ4D-R4Det_det3d_2x4_12e-final/epoch_2.pth` |
| 恢复训练 | `resume_from=None` |
| 图像分支 | `freeze_images=True`，并非所有参数都更新 |
| 输出 | 按配置保存 checkpoint、日志及相关可视化 |

## 15. 推理加载权重的代码证据

[tools/test_vod.py](../tools/test_vod.py) 第 265 行附近先构建模型：

```python
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
```

普通 checkpoint 分支在第 354 行附近加载参数：

```python
checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
```

之后测试流程调用模型，进入 `R4Det` 的推理路径。因此准确表述是：**训练入口组织训练并保存权重；推理入口构建模型并加载权重；R4Det.py 提供两阶段共用的模型实现。**
