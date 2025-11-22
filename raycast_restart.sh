#!/bin/bash

# WhisperBar Restart - Raycast Snippet
# Usage: Execute this script in Raycast to restart WhisperBar

cd /Users/salmenkhelifi/Documents/whisperbar_v1

# Stop if running
if pgrep -f "python.*main.py" > /dev/null; then
    pkill -f "python.*main.py"
    sleep 1
    
    # Force kill if still running
    if pgrep -f "python.*main.py" > /dev/null; then
        pkill -9 -f "python.*main.py"
    fi
    sleep 1
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
    osascript -e 'display notification "WhisperBar restarted successfully" with title "WhisperBar"'
else
    osascript -e 'display notification "Failed to restart. Check logs." with title "WhisperBar Error"'
    exit 1
fi

