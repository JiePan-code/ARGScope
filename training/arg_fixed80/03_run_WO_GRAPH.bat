@echo off
setlocal
chcp 65001 >nul
set "PYTHON_EXE=D:\Anaconda\envs\dnabert2_5060\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Python not found: %PYTHON_EXE%
  echo Edit PYTHON_EXE in this BAT to your CUDA-enabled Python path.
  pause
  exit /b 1
)
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
"%PYTHON_EXE%" -u "%~dp0run_pipeline.py" --variant wo_graph
set "RUN_EXIT=%ERRORLEVEL%"
if not "%RUN_EXIT%"=="0" echo STOPPED WITH ERROR. Read the message and logs; no old results were deleted.
if "%RUN_EXIT%"=="0" echo FINISHED SUCCESSFULLY.
pause
exit /b %RUN_EXIT%
