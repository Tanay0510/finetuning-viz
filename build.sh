#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run database migrations to update Neon
flask db upgrade
