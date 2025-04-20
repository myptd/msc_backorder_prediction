#!/bin/bash

# Create a new virtual environment named "bo"
python3.12 -m venv bo

# Activate the environment
source bo/bin/activate  # On macOS/Linux
# bo\Scripts\activate   # On Windows (if applicable)

# Upgrade pip
pip install --upgrade pip

# Install required libraries
pip install numpy pandas scikit-learn xgboost joblib matplotlib torch pytorch-lightning jupyterlab


# Verify installation
python -c "import numpy, pandas, sklearn, xgboost, joblib, matplotlib, torch, pytorch_lightning; print('All packages installed successfully')"

## additional libraries for specific tasks
pip install seaborn
pip install tensorboard
pip install imblearn

## # Install additional libraries for specific tasks
source bo/bin/activate
