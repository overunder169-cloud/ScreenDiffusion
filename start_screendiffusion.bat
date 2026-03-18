@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [Error] .venv\Scripts\python.exe not found.
  echo Run setup first in this folder.
  pause
  exit /b 1
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo Starting ScreenDiffusion...
".venv\Scripts\python.exe" ".\main_gpu_addon.py"

endlocal
