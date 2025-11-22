#!/bin/bash

# WhisperBar Stop Script
if pgrep -f "python.*main.py" > /dev/null; then
    pkill -f "python.*main.py"
    sleep 1
    
    # Force kill if still running
    if pgrep -f "python.*main.py" > /dev/null; then
        pkill -9 -f "python.*main.py"
    fi
    
    echo "✅ WhisperBar stopped"
else
    echo "ℹ️  WhisperBar is not running"
fi

