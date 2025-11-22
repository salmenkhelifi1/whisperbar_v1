#!/bin/bash

# Emergency force-kill script for WhisperBar
# Use this if the app is completely stuck or crashed

echo "🔨 Force killing all WhisperBar processes..."

# Kill all Python processes running main.py
pkill -9 -f "python.*main.py" 2>/dev/null
pkill -9 -f "main.py" 2>/dev/null

# Also kill by PID if found
ps aux | grep -i "python.*main.py" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null

sleep 1

# Check if anything is still running
if pgrep -f "python.*main.py" > /dev/null 2>&1; then
    echo "⚠️  Some processes may still be running. Try:"
    echo "   ps aux | grep main.py"
    echo "   kill -9 <PID>"
else
    echo "✅ All WhisperBar processes killed"
fi

