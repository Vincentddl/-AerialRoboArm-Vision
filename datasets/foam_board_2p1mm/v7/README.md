# Foam Board 2.1mm V7 数据集说明

本目录是当前 YOLO 泡沫板检测模型的主要训练/验证数据集。数据来自 2.1 mm 鱼眼镜头下的静态、动态、金属背景和 Z-axis 场景，目标类别只有一个：`foam_board`。

## 文件结构

| 路径 | 中文说明 |
| --- | --- |
| `foam_board_2p1mm.yaml` | Ultralytics YOLO 数据集配置文件，定义数据根目录、训练集、验证集和类别名称。 |
| `manifest.json` | 数据集构建清单，记录样本来源和版本信息，方便追溯。 |
| `images/train/` | 训练图片。图片通过 Git LFS 上传。 |
| `images/val/` | 验证图片。图片通过 Git LFS 上传。 |
| `labels/train/` | 训练标签，YOLO txt 格式。负样本标签文件可能为空，这是正常情况。 |
| `labels/val/` | 验证标签，YOLO txt 格式。负样本标签文件可能为空，这是正常情况。 |
| `debug/train/` | 训练集调试预览图，帮助人工检查样本选择和框标注。 |
| `debug/val/` | 验证集调试预览图，帮助人工检查样本选择和框标注。 |

## 类别定义

```text
0: foam_board
```

## 训练示例

```powershell
yolo detect train model=models/foam_board_2p1mm_v7.pt data=datasets/foam_board_2p1mm/v7/foam_board_2p1mm.yaml imgsz=512
```

如果协作者在其他路径克隆仓库，建议检查 `foam_board_2p1mm.yaml` 的 `path` 字段。必要时可以改成本机绝对路径或相对路径。

## 注意事项

- `.cache` 文件没有上传，Ultralytics 会在本地自动重建。
- 标签为空不代表文件损坏，通常表示该图片是负样本。
- 图片通过 Git LFS 管理，克隆后如果无法打开图片，请运行 `git lfs pull`。
