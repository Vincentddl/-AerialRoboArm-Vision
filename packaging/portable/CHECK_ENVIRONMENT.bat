@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Environment not installed.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import cv2, torch, ultralytics; print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__); print('OpenCV:', cv2.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
pause
