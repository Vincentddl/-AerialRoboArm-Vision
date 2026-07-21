# 输出文件说明

本目录只上传了协作者优化方向角预测、轨迹评估和目标输出复现所需的 JSON/JSONL、少量日志和轻量 bundle。大量图片、视频、接触表和历史中间结果仍保留在本地，没有上传。

## 方向角预测评估

| 文件 | 中文说明 |
| --- | --- |
| `bearing_prediction_eval_v7.json` | 当前主要方向角预测评估结果。使用 V7 检测、真实时间戳和鱼眼标定，评估保持角度、鲁棒匀角速度和鲁棒角加速度等方法。 |
| `bearing_prediction_eval_v7_unclipped.json` | 未裁剪条件下的 V7 方向角预测评估结果，用于和主评估做敏感性对比。 |
| `bearing_prediction_eval_validation_20260715.json` | 验证集方向角预测评估结果，用于检查方法是否只适配单一录像。 |

## 方向角估计训练记录

| 文件 | 中文说明 |
| --- | --- |
| `bearing_estimator_v1_training.json` | 本地训练 `bearing_estimator_v1.pt` 的详细指标和训练记录。 |
| `bearing_estimator_v1_remote_training.json` | 远程主机训练 `bearing_estimator_v1_remote.pt` 的详细指标和训练记录。 |

## 目标输出和轨迹评估

| 文件 | 中文说明 |
| --- | --- |
| `foam_board_targets.jsonl` | 历史 foam board 目标输出记录，用于检查目标格式和运行时输出口径。 |
| `merge_validation/v7_targets_0p4s.jsonl` | V7 在 0.4 s 预测设置下的目标输出记录，用于历史合并验证。 |
| `trajectory_0p5s_eval_20260714.json` | 0.5 s 轨迹预测评估摘要。 |
| `trajectory_2p1mm_test.jsonl` | 2.1 mm 镜头场景下的轨迹测试输出。 |
| `trajectory_lowconf_coast_test.jsonl` | 低置信度 coast 场景下的轨迹测试输出，用于检查丢检后的延续逻辑。 |
| `v6_20260714_002824_targets_eval.jsonl` | V6 目标输出评估记录。 |
| `v6_20260714_002824_targets_eval_cpu.jsonl` | V6 CPU 路径下的目标输出评估记录，用于排查设备差异。 |
| `v6_20260714_002824_targets_eval_full.jsonl` | V6 完整目标输出评估记录。 |
| `v7_zaxis_current_prediction_eval.json` | Z-axis 当前预测方案的历史评估摘要。 |
| `v7_zaxis_current_targets.jsonl` | Z-axis 场景下当前 V7 目标输出记录。 |

## 其他轻量文件

| 文件 | 中文说明 |
| --- | --- |
| `bearing_estimator_remote_bundle_v1.zip` | 远程训练返回的轻量 bundle，用于追溯 V1 方向角估计模型训练产物。 |
| `*.log` | 采集和补充数据时的标准输出/错误日志，体积很小，用于排查当时运行情况。 |

## 使用建议

- 优化方向角预测时，优先阅读 `bearing_prediction_eval_v7.json` 和 `docs/bearing_prediction_v7_evaluation.md`。
- 对比本地和远程 GRU 训练时，查看两个 `bearing_estimator_v1*_training.json`。
- 只优化 YOLO 检测模型时，本目录不是必需项；重点看 `datasets/foam_board_2p1mm/v7/` 和 `models/foam_board_2p1mm_v7.pt`。
