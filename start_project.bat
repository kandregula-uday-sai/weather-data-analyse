@echo off
setlocal
cd /d "%~dp0"
echo ==============================================
echo   Weather Data Analysis and Rain Prediction
echo ==============================================
where py >nul 2>nul
if errorlevel 1 (set PYTHON=python) else (set PYTHON=py)
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python environment...
  %PYTHON% -m venv .venv
  if errorlevel 1 goto failed
)
echo Installing project packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Starting website. Open http://127.0.0.1:5000
echo Keep this window open. Press Ctrl+C to stop.
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto failed
goto end
:failed
echo.
echo Something went wrong. Check the error above and read README.md.
pause
:end
endlocal
