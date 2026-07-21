# Foam Board 2.1mm Z-axis 轨迹数据说明

本目录保存 Z-axis 场景下的轨迹录像和检测结果，用于方向角预测、轨迹评估和离线回放。它不是普通 YOLO 训练集，而是用于验证“检测结果随时间变化时，能否预测目标未来方向角”的实验数据。

## 文件结构

| 路径 | 中文说明 |
| --- | --- |
| `raw/20260714_223253/` | 当前上传的 Z-axis 轨迹采集会话。 |
| `raw/20260714_223253/session.json` | 会话元信息，记录采集时间、数据来源和基本配置。 |
| `raw/20260714_223253/foam_board_2p1mm_zaxis_trajectory_20260714_223351_701282.mp4` | 原始轨迹录像，通过 Git LFS 上传。 |
| `raw/20260714_223253/foam_board_2p1mm_zaxis_trajectory_20260714_223351_701282.timestamps.jsonl` | 每帧真实时间戳，用于按真实时间间隔评估预测误差。 |
| `raw/20260714_223253/v6_predictions.json` | V6 模型在该录像上的检测结果。 |
| `raw/20260714_223253/v7_predictions.json` | V7 模型在该录像上的检测结果，当前方向角评估优先使用。 |
| `raw/20260714_223253/v7_evaluation.json` | V7 检测结果的评估摘要。 |
| `raw/20260714_223253/v7_candidate_selection.json` | V7 候选片段筛选记录。 |
| `raw/20260714_223253/*.jpg` | 候选片段、检测段和接触页预览图，通过 Git LFS 上传，用于人工检查。 |

## 离线回放示例

```powershell
& "D:\app\miniconda\envs\python312\python.exe" .\scripts\run_foam_board.py --source ".\datasets\foam_board_2p1mm_zaxis\raw\20260714_223253\foam_board_2p1mm_zaxis_trajectory_20260714_223351_701282.mp4"
```

## 方向角评估用途

方向角预测脚本会使用：

- V7 检测中心点。
- 鱼眼相机标定参数。
- 每帧真实时间戳。
- 只使用锚点之前的历史观测拟合未来方向角。

评估结果主要写入 `outputs/` 目录，具体说明见 `outputs/README.md`。

## 注意事项

- 录像和预览图通过 Git LFS 管理，克隆后需要运行 `git lfs pull`。
- 时间戳文件很重要，不能只看视频帧号；预测评估需要真实帧间隔。
- `v6_predictions.json` 主要用于历史对比，当前优先看 `v7_predictions.json`。
