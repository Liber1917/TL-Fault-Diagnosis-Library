@echo off
REM TL-Fault-Diagnosis-Library Environment Setup Script (Windows)
REM Usage: install_env.bat

echo ==========================================
echo TL-Fault-Diagnosis-Library Setup
echo ==========================================

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Conda detected, creating environment...
    conda env create -f environment.yml

    echo ==========================================
    echo Installation complete!
    echo Activate with: conda activate tl-fault
    echo ==========================================
    goto :end
)

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.8+ from python.org
    exit /b 1
)

python --version

REM Install PyTorch CPU version
echo Installing PyTorch (CPU)...
pip install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

REM Install other dependencies
echo Installing dependencies...
pip install -r requirements.txt

echo ==========================================
echo Installation complete!
echo ==========================================

:end
pause