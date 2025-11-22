#!/bin/bash

# WhisperBar Start - Raycast Snippet
# Usage: Execute this script in Raycast to start WhisperBar

cd /Users/salmenkhelifi/Documents/whisperbar_v1

# Check if already running
if pgrep -f "python.*main.py" > /dev/null; then
    osascript -e 'display notification "WhisperBar is already running" with title "WhisperBar"'
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
    osascript -e 'display notification "Python not found!" with title "WhisperBar Error"'
    exit 1
fi

# Start in background
nohup "$PYTHON_CMD" main.py > /tmp/whisperbar.log 2>&1 &
sleep 2

if pgrep -f "python.*main.py" > /dev/null; then
    osascript -e 'display notification "WhisperBar started successfully" with title "WhisperBar"'
else
    osascript -e 'display notification "Failed to start. Check logs." with title "WhisperBar Error"'
    exit 1
fi

