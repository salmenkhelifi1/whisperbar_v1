#!/bin/bash

# WhisperBar Stop - Raycast Snippet
# Usage: Execute this script in Raycast to stop WhisperBar

# Check if running
if pgrep -f "python.*main.py" > /dev/null; then
    pkill -f "python.*main.py"
    sleep 1
    
    # Force kill if still running
    if pgrep -f "python.*main.py" > /dev/null; then
        pkill -9 -f "python.*main.py"
    fi
    
    osascript -e 'display notification "WhisperBar stopped" with title "WhisperBar"'
else
    osascript -e 'display notification "WhisperBar is not running" with title "WhisperBar"'
fi

