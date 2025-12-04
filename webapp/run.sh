#!/bin/bash

# Startup script for No Limit Hold'em Web App

echo "Starting No Limit Texas Hold'em Web App..."
echo ""
echo "Make sure you have:"
echo "  1. Python 3.6+ installed"
echo "  2. Dependencies installed: pip install -r requirements.txt"
echo "  3. RLCard installed: cd ../rlcard && pip install -e ."
echo ""
echo "Starting Flask server on http://localhost:5001"
echo "Press Ctrl+C to stop"
echo ""

python3 app.py

