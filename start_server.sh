#!/bin/bash
# Script to start the face recognition server

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if requirements are installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Flask not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the server
echo "Starting face recognition server..."
python3 server.py
