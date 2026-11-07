@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run start_project.bat once first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" train.py
if errorlevel 1 pause
endlocal
