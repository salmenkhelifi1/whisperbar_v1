#!/bin/bash

# WhisperBar Launcher Script for Raycast
# This script can start or stop WhisperBar

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to find Python executable
find_python() {
    # Try venv first
    if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
        echo "venv/bin/python"
        return 0
    fi
    
    # Try python3
    if command -v python3 &> /dev/null; then
        echo "python3"
        return 0
    fi
    
    # Try python
    if command -v python &> /dev/null; then
        echo "python"
        return 0
    fi
    
    return 1
}

# Function to check if WhisperBar is running
is_running() {
    pgrep -f "python.*main.py" > /dev/null 2>&1
}

# Function to stop WhisperBar (forceful - handles crashes)
stop_app() {
    echo "🛑 Stopping WhisperBar..."
    
    # Try graceful kill first
    pkill -f "python.*main.py" 2>/dev/null
    sleep 2
    
    # Force kill if still running (handles crashes/stuck processes)
    if is_running; then
        echo "⚠️  Process still running, force killing..."
        pkill -9 -f "python.*main.py" 2>/dev/null
        sleep 1
    fi
    
    # Also kill any Python processes with main.py in the path
    pkill -9 -f "main.py" 2>/dev/null
    
    # Wait a moment and verify
    sleep 1
    if is_running; then
        echo "⚠️  Some processes may still be running. Trying alternative methods..."
        # Find and kill by PID
        ps aux | grep -i "python.*main.py" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
    fi
    
    if is_running; then
        echo "❌ Could not stop all processes. You may need to manually kill them."
        return 1
    else
        echo "✅ WhisperBar stopped"
        return 0
    fi
}

# Function to start WhisperBar
start_app() {
    if is_running; then
        echo "ℹ️  WhisperBar is already running"
        return 1
    fi
    
    # Find Python
    PYTHON_CMD=$(find_python)
    if [ $? -ne 0 ]; then
        echo "❌ Python not found!"
        exit 1
    fi
    
    # Check if main.py exists
    if [ ! -f "main.py" ]; then
        echo "❌ main.py not found!"
        exit 1
    fi
    
    # Activate venv if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    echo "🚀 Starting WhisperBar..."
    
    # Run in background
    nohup "$PYTHON_CMD" main.py > /tmp/whisperbar.log 2>&1 &
    
    sleep 2
    
    if is_running; then
        echo "✅ WhisperBar started (check menu bar for 🎤 icon)"
    else
        echo "❌ Failed to start WhisperBar. Check /tmp/whisperbar.log"
        exit 1
    fi
}

# Main logic
ACTION="${1:-start}"

case "$ACTION" in
    start)
        start_app
        ;;
    stop)
        stop_app
        ;;
    restart)
        stop_app
        sleep 2
        start_app
        ;;
    force-stop)
        echo "🔨 Force stopping all WhisperBar processes..."
        pkill -9 -f "python.*main.py" 2>/dev/null
        pkill -9 -f "main.py" 2>/dev/null
        ps aux | grep -i "python.*main.py" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
        sleep 1
        if is_running; then
            echo "❌ Some processes may still be running"
        else
            echo "✅ All processes force stopped"
        fi
        ;;
    status)
        if is_running; then
            echo "✅ WhisperBar is running"
        else
            echo "❌ WhisperBar is not running"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|force-stop}"
        echo ""
        echo "Commands:"
        echo "  start      - Start WhisperBar"
        echo "  stop       - Stop WhisperBar gracefully"
        echo "  restart    - Restart WhisperBar (stop then start)"
        echo "  status     - Check if WhisperBar is running"
        echo "  force-stop - Force kill all WhisperBar processes (use if stuck/crashed)"
        exit 1
        ;;
esac

