#!/bin/bash

# Stop all WhisperBar processes

echo "🛑 Stopping all WhisperBar processes..."

# Find and kill all Python processes running main.py
PIDS=$(pgrep -f "python.*main.py")

if [ -z "$PIDS" ]; then
    echo "ℹ️  No WhisperBar processes found"
else
    echo "Found processes: $PIDS"
    for PID in $PIDS; do
        echo "  Killing process $PID..."
        kill -TERM $PID 2>/dev/null
    done
    
    sleep 2
    
    # Force kill if still running
    REMAINING=$(pgrep -f "python.*main.py")
    if [ ! -z "$REMAINING" ]; then
        echo "Force killing remaining processes..."
        for PID in $REMAINING; do
            kill -9 $PID 2>/dev/null
        done
    fi
    
    echo "✅ All WhisperBar processes stopped"
fi

# Also kill any processes with whisperbar in the name
PIDS2=$(pgrep -f "whisperbar")
if [ ! -z "$PIDS2" ]; then
    echo "Found additional whisperbar processes: $PIDS2"
    for PID in $PIDS2; do
        kill -9 $PID 2>/dev/null
    done
fi

echo "✅ Done"

