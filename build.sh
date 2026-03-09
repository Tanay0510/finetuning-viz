#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run the database initialization
python init_db.py
