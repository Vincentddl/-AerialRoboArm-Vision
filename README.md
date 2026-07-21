# AerialRoboArm Vision

本仓库是 AerialRoboArm 的视觉识别与预测工程，包含相机标定、泡沫板 YOLO 检测、方向角预测评估、短期方向角估计训练，以及给机械臂运行时使用的模型权重。

当前仓库已经上传代码、配置、文档、测试、模型权重，以及给协作者优化模型所需的关键训练/评估数据。大体积历史输出和完整实验归档仍按 `.gitignore` 留在本地。

## 当前状态

- 相机：2.1 mm 鱼眼镜头，运行分辨率 `640x480`。
- 有效标定：`configs/camera_2p1mm_640x480_fisheye.json`。
- 标定重投影误差：约 `0.229 px`。
- 当前 YOLO 检测模型：`models/foam_board_2p1mm_v7.pt`。
- 当前方向角估计模型：`models/bearing_estimator_v1.pt`。
- 当前结论：方向角转换与离线评估流程已经完成，但现有录像上的 `400 ms` 外推尚未稳定超过保持角度基线，暂时不能直接用于抓取控制。

## 目录说明

| 路径 | 中文说明 |
| --- | --- |
| `.gitattributes` | Git LFS 规则。模型、数据集图片和视频通过 LFS 上传，避免普通 Git 历史膨胀。 |
| `.gitignore` | 本地数据、训练输出、缓存和历史归档的忽略规则。 |
| `requirements.txt` | Python 依赖版本要求。 |
| `calibration/` | 相机标定脚本、历史标定材料和本地调试图。部分原始图片仍被忽略。 |
| `configs/` | 正式运行使用的相机配置文件。 |
| `datasets/` | 已上传的关键训练集和 Z-axis 轨迹数据；详见 `datasets/README.md`。 |
| `docs/` | 训练记录、评估结论、历史说明和协作背景文档。 |
| `models/` | 已通过 Git LFS 上传的模型权重；详见 `models/README.md`。 |
| `outputs/` | 已上传的方向角、轨迹、目标输出和轻量日志；详见 `outputs/README.md`。 |
| `scripts/` | 数据集构建、离线评估、模型训练和实时运行入口脚本。 |
| `tests/` | 几何映射、预测器和估计器的单元测试。 |
| `tracking/` | 轨迹预测、方向角映射和坐标转换实现。 |
| `runs/` | 已上传 Ultralytics 训练参数、指标、曲线图和预览图；权重检查点仍不上传。 |
| `archive/` | 旧实验和无效采集归档，默认不上传。 |

## 协作者最常用文件

| 目标 | 必看文件 |
| --- | --- |
| 优化 YOLO 检测模型 | `datasets/foam_board_2p1mm/v7/README.md`、`datasets/foam_board_2p1mm/v7/foam_board_2p1mm.yaml`、`models/foam_board_2p1mm_v7.pt` |
| 优化方向角预测/轨迹评估 | `datasets/foam_board_2p1mm_zaxis/README.md`、`outputs/README.md`、`tracking/bearing.py`、`scripts/evaluate_bearing_prediction.py` |
| 复现短期方向角估计训练 | `scripts/build_bearing_estimation_dataset.py`、`scripts/train_bearing_estimator.py`、`docs/bearing_estimator_v1_training.md` |
| 检查项目完整性 | `scripts/project_doctor.py` |

## 环境准备

推荐使用本项目开发时的 Python 环境：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" --version
```

安装依赖：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" -m pip install -r requirements.txt
```

项目代码按 `Ultralytics 8.3.218` 整理。运行完整性检查：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" .\scripts\project_doctor.py
```

## YOLO 检测训练

当前已上传的主要训练集是：

```text
datasets/foam_board_2p1mm/v7/
```

数据集配置文件：

```text
datasets/foam_board_2p1mm/v7/foam_board_2p1mm.yaml
```

协作者可从当前 V7 权重继续训练或微调：

```powershell
yolo detect train model=models/foam_board_2p1mm_v7.pt data=datasets/foam_board_2p1mm/v7/foam_board_2p1mm.yaml imgsz=512
```

如果使用绝对路径训练，请先检查 `foam_board_2p1mm.yaml` 里的 `path` 是否符合自己的本地目录。

## 实时识别

使用摄像头实时运行：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" .\scripts\run_foam_board.py --source 0
```

使用已上传的 Z-axis 录像离线回放：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" .\scripts\run_foam_board.py --source ".\datasets\foam_board_2p1mm_zaxis\raw\20260714_223253\foam_board_2p1mm_zaxis_trajectory_20260714_223351_701282.mp4"
```

## 方向角预测离线评估

使用现有 V7 检测结果、真实帧时间戳和 2.1 mm 鱼眼标定参数运行：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" .\scripts\evaluate_bearing_prediction.py
```

默认结果写入：

```text
outputs/bearing_prediction_eval_v7.json
```

详细评估口径见：

```text
docs/bearing_prediction_v7_evaluation.md
```

## 短期方向角估计训练

先构建按完整轨迹分组的数据集：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" .\scripts\build_bearing_estimation_dataset.py
```

训练轻量 GRU 估计器：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" .\scripts\train_bearing_estimator.py --device cuda
```

当前 V1 模型只是伪标签实验模型，不会自动接入实时抓取。必须在独立人工标注轨迹上超过保持角度基线后，才能进入运行时集成。

## 测试

运行几何、预测器和估计器单元测试：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" -m unittest discover -s tests -v
```

## 数据上传策略

已经上传：

- `models/**/*.pt`
- `datasets/foam_board_2p1mm/v7/`
- `datasets/foam_board_2p1mm_zaxis/`
- 方向角、轨迹和目标相关的 `outputs/*.json` / `outputs/*.jsonl`
- `outputs/` 中少量日志和远程训练 bundle。
- `runs/` 中训练参数、指标 CSV、曲线图、混淆矩阵和 batch 预览图。

仍默认保留在本地：

- 完整历史 `outputs/` 图片、视频和中间结果。
- `runs/*/weights/*.pt` 训练检查点；当前关键权重已经单独整理到 `models/`。
- `archive/` 旧实验归档。
- 标定原始图片和大量调试角点图。

旧视觉运行说明保存在：

```text
docs/LEGACY_LAB_TRACKING_README.md
```
