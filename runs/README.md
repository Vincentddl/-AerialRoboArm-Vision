# Runs 训练输出说明

本目录保存 Ultralytics 训练过程输出。为了方便协作者分析训练质量，当前上传了训练参数、指标表、曲线图、混淆矩阵和 batch 预览图；没有上传 `weights/best.pt` 和 `weights/last.pt` 这类重复权重检查点。

## 已上传内容

| 文件类型 | 中文说明 |
| --- | --- |
| `args.yaml` | 本次训练使用的参数，适合复现训练设置。 |
| `results.csv` | 每个 epoch 的训练/验证指标，适合画曲线和比较收敛情况。 |
| `results.png` | Ultralytics 自动生成的训练指标总览图。 |
| `BoxF1_curve.png`、`BoxPR_curve.png`、`BoxP_curve.png`、`BoxR_curve.png` | 检测模型的 F1、PR、Precision、Recall 曲线。 |
| `confusion_matrix*.png` | 混淆矩阵图，用于查看误检/漏检情况。 |
| `labels.jpg` | 标签分布预览图。 |
| `train_batch*.jpg` | 训练 batch 预览图，用于检查增强和标注是否合理。 |
| `val_batch*_labels.jpg`、`val_batch*_pred.jpg` | 验证集标签和预测预览图，用于直观看模型效果。 |
| `comparison.json` | 个别历史实验的对比摘要。 |

## 未上传内容

| 路径 | 未上传原因 |
| --- | --- |
| `*/weights/best.pt` | 权重文件较大，且关键版本已整理上传到 `models/`。 |
| `*/weights/last.pt` | 训练末尾检查点较大，通常不是协作者分析模型的首要文件。 |

## 协作建议

- 当前重点看 `runs/foam_board_2p1mm_v7/`。
- 对比上一版效果时看 `runs/foam_board_2p1mm_v6/`。
- OOM 目录只用于追溯显存不足实验，不建议作为正式结果参考。
