#!/bin/bash

# WhisperBar Stop Script for Raycast

# Check if running
if pgrep -f "python.*main.py" > /dev/null; then
    echo "🛑 Stopping WhisperBar..."
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

