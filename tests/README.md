# 测试文件说明

本目录保存最小单元测试，用于防止方向角映射、预测器和短期估计模型的接口被改坏。

| 文件 | 中文说明 |
| --- | --- |
| `test_bearing.py` | 测试方向角映射、角度误差对称性、保持预测和鲁棒预测器对低置信度离群点的处理。 |
| `test_bearing_estimator.py` | 测试 GRU 方向角估计模型在可变长度输入下的前向输出形状。 |

## 运行方式

```powershell
& "D:\app\miniconda\envs\python312\python.exe" -m unittest discover -s tests -v
```

修改 `tracking/bearing.py` 或 `scripts/train_bearing_estimator.py` 后，建议至少运行这一组测试。
