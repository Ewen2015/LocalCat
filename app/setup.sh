#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run streamlit on demo.py with nohup
nohup streamlit run --server.port 20000 demo.py &