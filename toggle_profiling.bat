@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PROF_DIR=%CD%\profiling"
set "PID_FILE=%PROF_DIR%\profiling_pids.txt"

if not exist "%PROF_DIR%" mkdir "%PROF_DIR%"

if exist "%PID_FILE%" goto :stop
goto :start

:start
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%I"
set "GPU_CSV=%PROF_DIR%\gpu_metrics_%TS%.csv"
set "PROC_CSV=%PROF_DIR%\gpu_processes_%TS%.csv"

powershell -NoProfile -Command ^
  "$p1 = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem --format=csv -l 1 > ""%GPU_CSV%""' -WindowStyle Hidden -PassThru; " ^
  "$p2 = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_gpu_memory --format=csv -l 1 > ""%PROC_CSV%""' -WindowStyle Hidden -PassThru; " ^
  "Set-Content -Path '%PID_FILE%' -Value @($p1.Id, $p2.Id, 'GPU=%GPU_CSV%', 'PROC=%PROC_CSV%')"

if errorlevel 1 (
  echo Failed to start profiling.
  exit /b 1
)

echo Profiling started.
echo GPU log:  %GPU_CSV%
echo PROC log: %PROC_CSV%
echo Run this same file again to stop profiling.
exit /b 0

:stop
set "P1="
set "P2="
set "GPU_LOG="
set "PROC_LOG="

for /f "usebackq tokens=1* delims==" %%A in ("%PID_FILE%") do (
  if not defined P1 (
    set "P1=%%A"
  ) else if not defined P2 (
    set "P2=%%A"
  ) else (
    if /I "%%A"=="GPU" set "GPU_LOG=%%B"
    if /I "%%A"=="PROC" set "PROC_LOG=%%B"
  )
)

if defined P1 taskkill /PID %P1% /T /F >nul 2>nul
if defined P2 taskkill /PID %P2% /T /F >nul 2>nul

del /f /q "%PID_FILE%" >nul 2>nul

echo Profiling stopped.
if defined GPU_LOG echo GPU log:  %GPU_LOG%
if defined PROC_LOG echo PROC log: %PROC_LOG%
exit /b 0
