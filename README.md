# AerialRoboArm Vision

本目录将相机棋盘标定、泡沫块数据集、YOLO训练结果和轨迹实验整合为一个本地工程。原始目录仍保留，当前目录是后续开发入口。

## 当前状态

- 相机：2.1 mm鱼眼镜头，运行分辨率640x480。
- 棋盘：9x6内角点，方格边长20 mm。
- 有效标定：`configs/camera_2p1mm_640x480_fisheye.json`。
- 标定重投影误差：约0.229 px。
- 当前检测模型：`models/foam_board_2p1mm_v7.pt`。
- 当前轨迹模型：二维像素位置/速度/加速度Kalman，仅作为历史基线。
- 下一阶段：实现去畸变方向角的400 ms鲁棒预测。

## 目录

```text
calibration/  棋盘照片、标定脚本、历史标定结果和调试图
configs/      视觉运行使用的正式相机配置
datasets/     原始录像、图片、标签及V1-V7训练数据
models/       历史模型和当前V7权重
runs/         Ultralytics训练结果、指标图和检查点
outputs/      历史回放、预测输出和量化结果
scripts/      数据采集、数据集构建和运行入口
tracking/     当前轨迹与坐标映射实现
docs/         训练记录和旧版运行说明
archive/      旧实验与无效采集归档
```

## 环境

当前使用的解释器：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" --version
```

安装依赖：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" -m pip install -r requirements.txt
```

项目代码按Ultralytics 8.3.218整理。`scripts/project_doctor.py` 会报告实际环境版本是否一致。

## 完整性检查

```powershell
& "D:\app\miniconda\envs\python312\python.exe" .\scripts\project_doctor.py
```

## 棋盘采集与标定

采集新照片：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" .\calibration\capture_chessboard.py --source 0 --out .\calibration\images_new --cols 9 --rows 6 --width 640 --height 480
```

重新生成正式鱼眼参数：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" .\calibration\calibrate.py --images .\calibration\images_2p1mm_final --model fisheye --cols 9 --rows 6 --square-size 20 --out .\configs\camera_2p1mm_640x480_fisheye.json --debug-dir .\calibration\debug_active
```

相机、镜头、焦距或分辨率变化后必须重新标定。

## 实时识别

```powershell
& "D:\app\miniconda\envs\python312\python.exe" .\scripts\run_foam_board.py --source 0
```

离线回放示例：

```powershell
& "D:\app\miniconda\envs\python312\python.exe" .\scripts\run_foam_board.py --source ".\datasets\foam_board_2p1mm_zaxis\raw\20260714_223253\foam_board_2p1mm_zaxis_trajectory_20260714_223351_701282.mp4"
```

## 数据与Git

本地目录包含约7 GB完整实验资料。数据集、录像、训练运行目录和模型权重会保留在本地，但默认被`.gitignore`排除，避免普通Git提交超过GitHub限制。未来上传模型时使用Git LFS或GitHub Release，训练数据保留清单与版本说明。

旧视觉运行说明保存在`docs/LEGACY_LAB_TRACKING_README.md`。
