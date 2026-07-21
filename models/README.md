# 模型文件说明

本目录保存检测模型和方向角估计模型。所有 `.pt` 文件都通过 Git LFS 上传，克隆后需要确保 Git LFS 已安装并执行过 `git lfs pull`，否则本地只会得到指针文件。

## 当前推荐模型

| 文件 | 中文说明 |
| --- | --- |
| `foam_board_2p1mm_v7.pt` | 当前推荐的泡沫板 YOLO 检测模型。优先用于实时识别、离线回放和继续微调。 |
| `bearing_estimator_v1.pt` | 本地训练得到的短期方向角 GRU 估计器，当前只作为实验模型。 |
| `bearing_estimator_v1_remote.pt` | 远程主机训练得到的同版本方向角估计器，用于对比本地训练结果。 |

## 历史检测模型

| 文件 | 中文说明 |
| --- | --- |
| `foam_board_bootstrap_v1.pt` | 早期 bootstrap 检测模型，保留用于追溯数据和模型演进。 |
| `foam_board_bootstrap_v2.pt` | 第二版 bootstrap 检测模型，保留用于历史对比。 |
| `foam_board_2p1mm_v3_stage1.pt` | V3 第一阶段模型，用于追踪 V3 训练中间结果。 |
| `foam_board_2p1mm_v3.pt` | V3 完整模型，历史基线之一。 |
| `foam_board_2p1mm_v4.pt` | V4 检测模型，历史基线之一。 |
| `foam_board_2p1mm_v5.pt` | V5 检测模型，历史基线之一。 |
| `foam_board_2p1mm_v6.pt` | V6 检测模型，V7 之前的主要对照模型。 |

## 参考预训练模型

| 文件 | 中文说明 |
| --- | --- |
| `yolo11s.pt` | Ultralytics YOLO11s 参考权重，通常作为训练初始化或对照。 |
| `reference/yolo11n.pt` | YOLO11n 参考权重，小模型对照。 |
| `reference/yolo11s.duplicate-root-copy.pt` | `yolo11s.pt` 的重复参考副本，保留用于历史路径兼容。 |
| `reference/yolov8n.pt` | YOLOv8n 参考权重，用于早期实验或轻量对照。 |

## 使用建议

- 继续优化检测模型时，优先从 `foam_board_2p1mm_v7.pt` 开始。
- 方向角估计模型不要直接接入真实抓取控制，先用独立人工标注轨迹验证。
- 如果朋友克隆后模型文件只有几 KB，说明没有拉取 LFS 对象，需要运行 `git lfs pull`。
