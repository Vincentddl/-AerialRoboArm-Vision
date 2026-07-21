# AerialRoboArm Vision

本仓库是 AerialRoboArm 项目的视觉识别与预测工程，主要用于识别泡沫板目标、输出目标位置/方向信息，并为后续机械臂抓取控制提供视觉输入。

仓库内容包括：

- 2.1 mm 鱼眼相机标定配置。
- 泡沫板 YOLO 检测模型和训练数据。
- Z-axis 场景轨迹录像、检测结果和方向角预测评估数据。
- YOLO 训练记录、曲线图、混淆矩阵和 batch 预览图。
- 方向角预测、短期 GRU 估计器和相关测试。
- 给协作者阅读的中文说明文档。

当前仓库已经尽量上传了协作者优化模型所需的核心内容。过大的历史视频、完整中间输出、训练检查点重复权重、缓存文件仍保留在本地，不作为默认协作内容上传。

## 快速结论

| 项目 | 当前状态 |
| --- | --- |
| 当前推荐检测模型 | `models/foam_board_2p1mm_v7.pt` |
| 当前推荐训练集 | `datasets/foam_board_2p1mm/v7/` |
| 当前相机配置 | `configs/camera_2p1mm_640x480_fisheye.json` |
| 当前方向角估计模型 | `models/bearing_estimator_v1.pt` |
| 当前方向角结论 | 100 ms / 200 ms 可继续实验，400 ms 外推尚未可靠超过保持角度基线 |
| 模型和媒体存储方式 | Git LFS |
| 主要运行环境 | Python 3.12、Ultralytics 8.3.218、OpenCV、NumPy |

## 给协作者的接手步骤

1. 克隆仓库。

```powershell
git clone https://github.com/Vincentddl/-AerialRoboArm-Vision.git
cd -AerialRoboArm-Vision
```

2. 拉取 Git LFS 文件。

```powershell
git lfs install
git lfs pull
```

如果没有执行 `git lfs pull`，模型、图片和视频可能只是几 KB 的指针文件，无法正常训练或打开。

3. 安装依赖。

```powershell
python -m pip install -r requirements.txt
```

本项目开发时使用的解释器是：

```powershell
D:\app\miniconda\envs\python312\python.exe
```

朋友电脑路径不同是正常的，可以使用自己的 Conda/venv 环境。

4. 运行完整性检查。

```powershell
python scripts/project_doctor.py
```

5. 跑单元测试。

```powershell
python -m unittest discover -s tests -v
```

## 目录总览

| 路径 | 内容说明 | 是否重点关注 |
| --- | --- | --- |
| `.gitattributes` | Git LFS 跟踪规则。`.pt`、数据集图片/视频、runs 图片、部分 npy 通过 LFS 管理。 | 是 |
| `.gitignore` | 忽略规则。大数据、缓存、历史视频、训练权重重复件默认不提交。 | 是 |
| `requirements.txt` | Python 依赖。 | 是 |
| `configs/` | 正式相机配置，尤其是 `camera_2p1mm_640x480_fisheye.json`。 | 是 |
| `models/` | 已上传的模型权重，包含当前 V7 检测模型和方向角估计模型。 | 是 |
| `datasets/foam_board_2p1mm/v7/` | 当前 YOLO 检测训练集。 | 是 |
| `datasets/foam_board_2p1mm_zaxis/` | Z-axis 轨迹录像、时间戳和 V6/V7 检测结果。 | 是 |
| `outputs/` | 方向角、轨迹、目标输出、日志和轻量 bundle。 | 是 |
| `runs/` | YOLO 训练参数、指标、曲线图、混淆矩阵和 batch 预览图。 | 是 |
| `scripts/` | 数据采集、数据集构建、训练、评估和运行脚本。 | 是 |
| `tracking/` | 轨迹预测和方向角映射实现。 | 是 |
| `tests/` | 单元测试。 | 是 |
| `docs/` | 实验说明、训练记录、评估结论和历史文档。 | 参考 |
| `calibration/` | 标定脚本、历史标定配置和少量清单；大量原始图片未上传。 | 参考 |
| `archive/` | 少量历史脚本和无效采集记录；大视频未上传。 | 参考 |

## 已上传和未上传内容

### 已上传

| 类型 | 路径 | 用途 |
| --- | --- | --- |
| 项目代码 | `scripts/`、`tracking/`、`tests/` | 运行、训练、评估和测试 |
| 相机配置 | `configs/camera_2p1mm_640x480_fisheye.json` | 鱼眼去畸变和方向角映射 |
| 检测模型 | `models/foam_board_2p1mm_v7.pt` | 当前主模型 |
| 历史检测模型 | `models/foam_board_2p1mm_v3.pt` 到 `v6.pt` 等 | 版本对比 |
| 方向角模型 | `models/bearing_estimator_v1.pt`、`models/bearing_estimator_v1_remote.pt` | 短期方向角估计实验 |
| YOLO V7 数据集 | `datasets/foam_board_2p1mm/v7/` | 继续训练或微调检测模型 |
| Z-axis 数据 | `datasets/foam_board_2p1mm_zaxis/` | 离线回放和方向角预测评估 |
| 评估输出 | `outputs/*.json`、`outputs/*.jsonl` | 复现预测评估和目标输出分析 |
| 训练记录 | `runs/` 中的 `args.yaml`、`results.csv`、曲线图、混淆矩阵、batch 预览图 | 分析模型训练质量 |
| 中文说明 | 各目录 `README.md` | 让协作者快速理解文件用途 |

### 暂未上传

| 类型 | 原因 |
| --- | --- |
| `outputs/` 中大量视频、图片和历史中间结果 | 总量约数 GB，多数可再生成，不适合默认上传 |
| `runs/*/weights/best.pt` 和 `runs/*/weights/last.pt` | 训练检查点较大，且关键权重已经整理到 `models/` |
| 完整历史数据集 | 当前模型优化优先使用 V7 数据集，历史数据会显著增加仓库体积 |
| 标定原始图片和 debug 角点图 | 数量多、体积大；正式配置 JSON 已上传 |
| `__pycache__/`、`*.pyc`、`.cache` | 自动生成缓存，不应提交 |
| 无效采集大视频 | 主要用于本地追溯，不是当前优化必要输入 |

## Git LFS 说明

以下文件通过 Git LFS 管理：

```text
models/*.pt
models/**/*.pt
datasets/**/*.jpg
datasets/**/*.mp4
runs/**/*.jpg
runs/**/*.png
assets/**/*.npy
```

协作者克隆后如果模型或图片无法打开，先运行：

```powershell
git lfs pull
```

检查 LFS 文件：

```powershell
git lfs ls-files
```

## 环境和依赖

`requirements.txt` 当前内容：

```text
numpy>=1.26,<3
opencv-python>=4.10,<5
ultralytics==8.3.218
```

如果需要训练方向角 GRU，代码还会用到 PyTorch。当前本地训练使用过的环境包含 CUDA 版 PyTorch；朋友可以根据自己的 GPU 安装对应版本。

检查项目关键文件：

```powershell
python scripts/project_doctor.py
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

## YOLO 检测模型优化

### 关键文件

| 文件 | 用途 |
| --- | --- |
| `models/foam_board_2p1mm_v7.pt` | 当前推荐 YOLO 检测权重 |
| `datasets/foam_board_2p1mm/v7/foam_board_2p1mm.yaml` | YOLO 数据集配置 |
| `datasets/foam_board_2p1mm/v7/images/train/` | 训练图片 |
| `datasets/foam_board_2p1mm/v7/images/val/` | 验证图片 |
| `datasets/foam_board_2p1mm/v7/labels/train/` | 训练标签 |
| `datasets/foam_board_2p1mm/v7/labels/val/` | 验证标签 |
| `runs/foam_board_2p1mm_v7/` | V7 训练指标、曲线和预览图 |

### 模型来源和历史权重

本仓库不是只有最终模型，已经上传了从 bootstrap 到 V7 的主要检测权重，方便朋友做横向对比、回退测试或者继续微调。

| 权重文件 | 建议用途 |
| --- | --- |
| `models/foam_board_2p1mm_v7.pt` | 当前主模型，优先用于实时识别、离线回放、继续训练和效果评估。 |
| `models/foam_board_2p1mm_v6.pt` | V7 前一版主模型，适合在怀疑 V7 过拟合或轨迹逻辑异常时做对照。 |
| `models/foam_board_2p1mm_v5.pt` | 历史基线，可用于确认 V6/V7 改动是否真的改善了动态场景。 |
| `models/foam_board_2p1mm_v4.pt` | 更早的 2.1 mm 镜头模型，主要用于追溯训练演进。 |
| `models/foam_board_2p1mm_v3.pt` | 早期完整模型，可作为最低限度回退参考。 |
| `models/foam_board_2p1mm_v3_stage1.pt` | V3 第一阶段中间模型，通常不建议作为当前运行模型。 |
| `models/foam_board_bootstrap_v1.pt`、`models/foam_board_bootstrap_v2.pt` | 早期 bootstrap 模型，主要用于历史追溯，不作为当前推荐入口。 |
| `models/yolo11s.pt`、`models/reference/*.pt` | Ultralytics 预训练/参考权重，用于重新训练或模型结构对照。 |

如果朋友只想优化当前检测效果，优先从 `foam_board_2p1mm_v7.pt` 和 `datasets/foam_board_2p1mm/v7/` 开始，不需要先重建 V3 到 V6。

### 数据集类别

```text
0: foam_board
```

### 训练命令示例

在仓库根目录运行：

```powershell
yolo detect train model=models/foam_board_2p1mm_v7.pt data=datasets/foam_board_2p1mm/v7/foam_board_2p1mm.yaml imgsz=512
```

如果朋友电脑上 `foam_board_2p1mm.yaml` 的 `path` 指向原作者本地路径，可以改为自己的绝对路径，或者改成相对路径。

### 评估/预测命令示例

```powershell
yolo detect val model=models/foam_board_2p1mm_v7.pt data=datasets/foam_board_2p1mm/v7/foam_board_2p1mm.yaml imgsz=512
```

```powershell
yolo detect predict model=models/foam_board_2p1mm_v7.pt source=datasets/foam_board_2p1mm/v7/images/val imgsz=512
```

### 临时切换旧模型运行

`scripts/run_foam_board.py` 默认使用 `models/foam_board_2p1mm_v7.pt`。如果想临时测试旧模型，不需要改代码，直接在命令后覆盖 `--model` 参数即可：

```powershell
python scripts/run_foam_board.py --source 0 --device cpu --model models/foam_board_2p1mm_v6.pt
```

或者对已上传的 Z-axis 录像离线回放：

```powershell
python scripts/run_foam_board.py --source "datasets/foam_board_2p1mm_zaxis/raw/20260714_223253/foam_board_2p1mm_zaxis_trajectory_20260714_223351_701282.mp4" --device cpu --model models/foam_board_2p1mm_v6.pt
```

这类回退测试的好处是可以快速判断问题来自检测模型本身，还是来自后面的轨迹预测、方向角映射或控制逻辑。缺点是旧模型识别精度和新场景适配性通常弱于 V7，不能把旧模型跑通误认为整体方案已经可靠。

### 看训练结果时重点看什么

| 文件 | 观察重点 |
| --- | --- |
| `runs/foam_board_2p1mm_v7/results.csv` | loss 是否下降、precision/recall/mAP 是否稳定 |
| `runs/foam_board_2p1mm_v7/results.png` | 整体训练曲线 |
| `runs/foam_board_2p1mm_v7/BoxPR_curve.png` | PR 曲线形状 |
| `runs/foam_board_2p1mm_v7/confusion_matrix.png` | 是否存在明显误检/漏检 |
| `runs/foam_board_2p1mm_v7/val_batch*_pred.jpg` | 验证集预测可视化 |
| `runs/foam_board_2p1mm_v7/train_batch*.jpg` | 数据增强和标签是否合理 |

## 实时识别和离线回放

实时摄像头运行：

```powershell
python scripts/run_foam_board.py --source 0
```

使用已上传的 Z-axis 录像离线回放：

```powershell
python scripts/run_foam_board.py --source "datasets/foam_board_2p1mm_zaxis/raw/20260714_223253/foam_board_2p1mm_zaxis_trajectory_20260714_223351_701282.mp4"
```

`scripts/run_foam_board.py` 是推荐入口；更底层的 YOLO 检测、跟踪和输出逻辑在 `scripts/yolo_track.py`。

## 方向角预测和轨迹评估

方向角预测的目标是：把像素检测中心通过相机标定映射成去畸变后的方向角，然后根据历史观测预测短期未来方向。

### 关键文件

| 文件 | 用途 |
| --- | --- |
| `tracking/bearing.py` | 方向角映射、角度误差、保持预测和鲁棒预测器 |
| `scripts/evaluate_bearing_prediction.py` | 离线评估入口 |
| `datasets/foam_board_2p1mm_zaxis/` | 录像、时间戳、V6/V7 检测结果 |
| `outputs/bearing_prediction_eval_v7.json` | 当前主评估结果 |
| `docs/bearing_prediction_v7_evaluation.md` | 评估口径和结论说明 |

### 运行评估

```powershell
python scripts/evaluate_bearing_prediction.py
```

默认输出：

```text
outputs/bearing_prediction_eval_v7.json
```

### 当前结论

- 方向角映射流程已建立。
- 100 ms 和 200 ms 预测可作为短期实验方向。
- 400 ms 外推在当前数据上尚未稳定超过保持角度基线。
- 当前模型不能直接驱动真实抓取控制，需要进一步标注和验证。

## 短期方向角 GRU 估计器

### 构建训练数据

```powershell
python scripts/build_bearing_estimation_dataset.py
```

### 训练模型

```powershell
python scripts/train_bearing_estimator.py --device cuda
```

如果没有 GPU，可以改用：

```powershell
python scripts/train_bearing_estimator.py --device cpu
```

### 相关文件

| 文件 | 用途 |
| --- | --- |
| `models/bearing_estimator_v1.pt` | 本地训练的 V1 方向角估计模型 |
| `models/bearing_estimator_v1_remote.pt` | 远程主机训练的 V1 对照模型 |
| `outputs/bearing_estimator_v1_training.json` | 本地训练报告 |
| `outputs/bearing_estimator_v1_remote_training.json` | 远程训练报告 |
| `docs/bearing_estimator_v1_training.md` | 训练过程和结论说明 |

## 相机标定

当前正式标定文件：

```text
configs/camera_2p1mm_640x480_fisheye.json
```

历史标定文件和脚本保存在：

```text
calibration/
```

相机、镜头、焦距或运行分辨率变化后，必须重新标定，否则方向角映射会失真。

采集棋盘图片示例：

```powershell
python calibration/capture_chessboard.py --source 0 --out calibration/images_new --cols 9 --rows 6 --width 640 --height 480
```

重新标定示例：

```powershell
python calibration/calibrate.py --images calibration/images_2p1mm_final --model fisheye --cols 9 --rows 6 --square-size 20 --out configs/camera_2p1mm_640x480_fisheye.json --debug-dir calibration/debug_active
```

注意：当前仓库没有上传大量原始棋盘图片和 debug 角点图，只上传了正式配置和部分历史 JSON。

## 常用脚本说明

| 脚本 | 用途 |
| --- | --- |
| `scripts/project_doctor.py` | 检查关键文件是否齐全 |
| `scripts/run_foam_board.py` | 当前推荐实时/离线运行入口 |
| `scripts/yolo_track.py` | YOLO 检测、跟踪、目标输出底层逻辑 |
| `scripts/build_foam_board_2p1mm_v7.py` | 构建 V7 数据集的历史脚本 |
| `scripts/evaluate_bearing_prediction.py` | 方向角预测离线评估 |
| `scripts/build_bearing_estimation_dataset.py` | 构建方向角估计训练数据 |
| `scripts/train_bearing_estimator.py` | 训练方向角 GRU 估计器 |
| `scripts/capture_dataset.py` | 采集实验数据 |
| `scripts/camera_latency_test.py` | 测试相机延迟 |

更详细的脚本说明见：

```text
scripts/README.md
```

## 常见问题

### 模型文件只有几 KB

这是 Git LFS 对象没有拉下来。运行：

```powershell
git lfs pull
```

### YOLO 找不到数据集

检查：

```text
datasets/foam_board_2p1mm/v7/foam_board_2p1mm.yaml
```

里面的 `path` 可能是原作者本地绝对路径。朋友需要改成本机路径，或者改成相对路径。

### 训练结果里没有 weights

这是有意的。`runs/*/weights/*.pt` 没有上传，因为关键权重已整理到 `models/`，避免重复占用 LFS 空间。

### 方向角评估结果不如保持角度

这是当前已知结论，不是运行错误。现有数据上 400 ms 外推还不可靠，后续需要更多人工标注轨迹和更稳定的预测方法。

### 缺少很多 outputs 图片/视频

这是有意的。`outputs/` 中大量历史图片和视频体积很大，不是朋友优化模型的必要输入。当前上传的是 JSON/JSONL、日志和轻量 bundle。

## 推荐协作路线

### 如果目标是优化 YOLO 检测

1. 先用 `models/foam_board_2p1mm_v7.pt` 跑验证集。
2. 查看 `runs/foam_board_2p1mm_v7/` 的曲线和预测图。
3. 检查 `datasets/foam_board_2p1mm/v7/debug/` 的样本质量。
4. 根据误检/漏检补数据或调训练参数。
5. 训练新模型后，把新权重放到 `models/`，并用 Git LFS 上传。

### 如果目标是优化方向角预测

1. 阅读 `tracking/bearing.py`。
2. 运行 `scripts/evaluate_bearing_prediction.py`。
3. 对比 `outputs/bearing_prediction_eval_v7.json`。
4. 修改预测器或数据构造方式。
5. 运行 `tests/test_bearing.py` 和完整单元测试。

### 如果目标是接入机械臂控制

1. 先不要直接使用 400 ms 外推。
2. 优先验证 100 ms / 200 ms 短期预测。
3. 使用独立人工标注数据验证。
4. 只有超过保持角度基线后，再考虑接入运行时控制。

## 参考文档

| 文件 | 内容 |
| --- | --- |
| `docs/bearing_prediction_v7_evaluation.md` | V7 方向角预测评估口径和结论 |
| `docs/bearing_estimator_v1_training.md` | V1 GRU 方向角估计器训练记录 |
| `docs/LOCAL_ARCHIVE_INVENTORY.md` | 本地完整资料清单和哪些未上传 |
| `docs/LEGACY_LAB_TRACKING_README.md` | 旧版视觉运行说明 |

## 当前维护原则

- 代码、配置、说明、测试尽量上传。
- 对协作者有价值的轻量实验记录尽量上传。
- 模型、图片、视频、npy 等二进制文件走 Git LFS。
- 原始大视频、可再生成中间图、重复训练权重和缓存默认不上传。
- 新模型进入 `models/` 后，要同步更新 `models/README.md` 和根 README。
