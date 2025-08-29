#!/bin/bash

# Exit immediately if a command fails
set -e

# 1. Create virtual environment
python3 -m venv req_grp

# 2. Activate environment
source req_grp/bin/activate

# 3. Upgrade pip (optional but recommended)
pip install --upgrade pip

# 4. Install dependencies from requirements.txt
pip install -r lib.txt

# 5. Run your main script
python runner.py
