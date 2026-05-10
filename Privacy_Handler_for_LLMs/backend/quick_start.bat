@echo off
echo ========================================
echo PII Privacy Handler Backend - Quick Start
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo [2/3] Installing required packages...
pip install flask flask-cors requests google-generativeai 2>nul

echo.
echo [3/3] Starting backend server...
echo.
echo Backend will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
echo ========================================
python app.py
