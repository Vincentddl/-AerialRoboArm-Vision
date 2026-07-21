# 脚本文件说明

本目录保存数据采集、数据集构建、模型训练、离线评估和实时运行入口。优先使用根 README 中给出的命令；下面是每个脚本的用途说明。

| 文件 | 中文说明 |
| --- | --- |
| `run_foam_board.py` | 当前推荐的泡沫板实时/离线运行入口，封装默认模型、相机标定和预测参数。 |
| `yolo_track.py` | 底层 YOLO 检测、目标跟踪、预测输出和结果保存脚本。`run_foam_board.py` 会调用或复用它的逻辑。 |
| `evaluate_bearing_prediction.py` | 方向角预测离线评估脚本，使用 V7 检测结果、真实时间戳和鱼眼标定计算预测误差。 |
| `build_bearing_estimation_dataset.py` | 从 V7 检测结果和时间戳构建短期方向角估计训练数据。 |
| `train_bearing_estimator.py` | 训练轻量 GRU 方向角估计器，输出 `models/bearing_estimator_v1.pt` 和训练报告。 |
| `project_doctor.py` | 项目完整性检查脚本，检查关键模型、数据、配置和脚本是否存在。 |
| `capture_dataset.py` | 摄像头数据采集脚本，用于采集泡沫板实验图像或录像材料。 |
| `camera_latency_test.py` | 相机延迟测试脚本，用于估计采集链路延迟。 |
| `build_foam_board_bootstrap.py` | 早期 bootstrap 数据集构建脚本，保留用于历史追溯。 |
| `build_foam_board_v2.py` | V2 数据集构建脚本，保留用于历史追溯。 |
| `build_foam_board_2p1mm_v3.py` | 2.1 mm 镜头 V3 数据集构建脚本。 |
| `build_foam_board_2p1mm_v3_final.py` | V3 final 数据集构建脚本，保留用于历史复现。 |
| `build_foam_board_2p1mm_v4.py` | V4 数据集构建脚本。 |
| `build_foam_board_2p1mm_v5.py` | V5 数据集构建脚本。 |
| `build_foam_board_2p1mm_v6.py` | V6 数据集构建脚本。 |
| `build_foam_board_2p1mm_v7.py` | 当前 V7 数据集构建脚本，和已上传的 `datasets/foam_board_2p1mm/v7/` 对应。 |
| `run_lab.py` | 旧版实验室运行入口，当前主要保留兼容和历史参考。 |

## 协作者建议

- 只优化 YOLO 检测模型时，优先看 `run_foam_board.py`、`yolo_track.py` 和 `build_foam_board_2p1mm_v7.py`。
- 优化方向角预测时，优先看 `evaluate_bearing_prediction.py`、`build_bearing_estimation_dataset.py` 和 `train_bearing_estimator.py`。
- 修改脚本后建议运行 `python -m unittest discover -s tests -v`。
