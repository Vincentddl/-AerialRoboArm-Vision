@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

py -3.9 -c "import sys; print(sys.version)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.9 x64 was not found.
    echo Install Python 3.9 x64 and enable the Python launcher, then retry.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3.9 -m venv .venv
    if errorlevel 1 goto :failed
)

call ".venv\Scripts\activate.bat"
echo Installing CPU-only PyTorch...
python -m pip install --upgrade pip
if errorlevel 1 goto :failed
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :failed
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Installation finished. CPU inference is supported but will be slower.
echo Run RUN_PROJECT.bat to start.
pause
exit /b 0

:failed
echo.
echo [ERROR] Installation failed. Check the network connection and messages above.
pause
exit /b 1
