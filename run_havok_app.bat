@echo off
REM ============================================
REM  HAVOK Regime-Shift Detector — One-Click App
REM  Double-click this file to launch.
REM ============================================

echo Starting HAVOK Regime-Shift Detector...
echo.

REM Try to find Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Install Python from https://python.org
    pause
    exit /b 1
)

REM Launch the Streamlit app
python -m streamlit run havolib/dashboard/simple.py --server.headless true

pause
