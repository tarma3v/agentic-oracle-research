#!/bin/bash
# Setup script for virtual environment

set -e

echo "Setting up virtual environment..."

# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install project and test dependencies
pip install -e ".[dev]"

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the venv, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then test Exa integration with:"
echo "  python test_exa.py"
