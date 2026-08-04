@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] The environment is not installed.
    echo Run INSTALL_GPU.bat for an NVIDIA GPU, or INSTALL_CPU.bat for CPU-only use.
    pause
    exit /b 1
)

if not exist "models\foam_board_2p1mm_v7.pt" (
    echo [ERROR] Model file models\foam_board_2p1mm_v7.pt is missing.
    pause
    exit /b 1
)

set "YOLO_CONFIG_DIR=%CD%\outputs\ultralytics"
if not exist "outputs\ultralytics" mkdir "outputs\ultralytics"

".venv\Scripts\python.exe" -u "scripts\run_foam_board.py" --source 0
if errorlevel 1 (
    echo.
    echo [ERROR] The project stopped with an error. Review the messages above.
    pause
)
