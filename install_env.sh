#!/bin/bash
# TL-Fault-Diagnosis-Library Environment Setup Script
# Usage: bash install_env.sh

set -e

echo "=========================================="
echo "TL-Fault-Diagnosis-Library Setup"
echo "=========================================="

# Detect OS
OS="$(uname -s)"
echo "Detected OS: $OS"

# Create conda environment if conda is available
if command -v conda &> /dev/null; then
    echo "Conda detected, creating environment..."

    # Create environment from yml
    conda env create -f environment.yml

    echo "=========================================="
    echo "Installation complete!"
    echo "Activate with: conda activate tl-fault"
    echo "=========================================="

else
    echo "Conda not detected, using pip..."

    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "Python version: $PYTHON_VERSION"

    # Create virtual environment (optional)
    if [ "$1" = "--venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
    fi

    # Install PyTorch CPU version
    echo "Installing PyTorch (CPU)..."
    pip install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

    # Install other dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt

    echo "=========================================="
    echo "Installation complete!"
    if [ "$1" = "--venv" ]; then
        echo "Activate with: source venv/bin/activate"
    fi
    echo "=========================================="
fi