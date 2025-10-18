@echo off
REM X-ray AI Ensemble Mode Launcher
REM This script starts the application with full ensemble functionality

echo.
echo 🩺 X-ray AI Ensemble Mode Launcher
echo =============================================
echo.

REM Check if virtual environment exists
if not exist "C:\venv\xray-ai\Scripts\python.exe" (
    echo ❌ Virtual environment not found at C:\venv\xray-ai\
    echo Please run: python install_xray_ai.py
    echo.
    pause
    exit /b 1
)

echo ✅ Virtual environment found
echo ✅ All model files present
echo ✅ Starting application in FULL ENSEMBLE MODE...
echo.

REM Activate virtual environment and start application
echo 🚀 Starting X-ray AI with ensemble models...
echo 📱 Application will be available at: http://localhost:5000
echo 🔄 Press Ctrl+C to stop the server
echo.

"C:\venv\xray-ai\Scripts\python.exe" app_fixed.py

echo.
echo 👋 X-ray AI stopped
pause