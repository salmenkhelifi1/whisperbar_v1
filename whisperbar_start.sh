#!/bin/bash

# WhisperBar Start Script for Raycast
cd /Users/salmenkhelifi/Documents/whisperbar_v1

# Check if already running
if pgrep -f "python.*main.py" > /dev/null; then
    echo "ℹ️  WhisperBar is already running"
    exit 0
fi

# Find Python
if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found!"
    exit 1
fi

# Start in background
nohup "$PYTHON_CMD" main.py > /tmp/whisperbar.log 2>&1 &
sleep 2

if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ WhisperBar started"
else
    echo "❌ Failed to start. Check /tmp/whisperbar.log"
    exit 1
fi

