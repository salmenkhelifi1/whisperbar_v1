#!/bin/bash

# WhisperBar Quick Start - Install and Launch
# This script will install dependencies and launch the app

set -e  # Exit on any error

# Fix PATH for GUI-launched terminals (Raycast, etc.)
# Add common Homebrew paths
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Source common shell profiles to get proper PATH
if [ -f ~/.zshrc ]; then
    source ~/.zshrc 2>/dev/null || true
elif [ -f ~/.bash_profile ]; then
    source ~/.bash_profile 2>/dev/null || true
elif [ -f ~/.bashrc ]; then
    source ~/.bashrc 2>/dev/null || true
fi

echo "🎤 WhisperBar Quick Start"
echo "========================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Function to check if virtual environment exists and is valid
check_venv() {
    if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
        source venv/bin/activate
        if python -c "import torch, transformers, sounddevice, rumps" 2>/dev/null; then
            return 0  # venv exists and has required packages
        else
            echo "🔄 Virtual environment exists but missing packages. Reinstalling..."
            deactivate 2>/dev/null || true
            rm -rf venv
            return 1
        fi
    else
        return 1  # venv doesn't exist
    fi
}

# Check if we need to install
if ! check_venv; then
    echo "📦 Setting up WhisperBar (first time setup)..."
    echo ""
    
    # Check Python version
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
        echo "❌ Error: Python 3.9+ is required"
        echo "Please install Python 3.9+ from https://python.org"
        exit 1
    fi
    
    # Check if we're on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        echo "❌ Error: This app only works on macOS"
        exit 1
    fi
    
    # Create virtual environment
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    echo "📥 Installing dependencies (this may take a few minutes)..."
    pip install -r requirements.txt
    
    # Check for ffmpeg and install if needed
    echo "🔧 Checking for ffmpeg..."
    
    # Check multiple possible locations
    FFMPEG_FOUND=false
    for path in "/opt/homebrew/bin/ffmpeg" "/usr/local/bin/ffmpeg" "$(which ffmpeg 2>/dev/null)"; do
        if [ -x "$path" ]; then
            echo "✅ ffmpeg found at: $path"
            FFMPEG_FOUND=true
            break
        fi
    done
    
    if [ "$FFMPEG_FOUND" = false ]; then
        echo "📦 Installing ffmpeg (required for audio processing)..."
        
        # Try multiple package managers
        if command -v brew &> /dev/null; then
            echo "   Using Homebrew..."
            brew install ffmpeg
        elif command -v port &> /dev/null; then
            echo "   Using MacPorts..."
            sudo port install ffmpeg
        else
            echo "⚠️  No package manager found."
            echo "   Please install one of:"
            echo "   • Homebrew: https://brew.sh then run: brew install ffmpeg"
            echo "   • MacPorts: https://macports.org then run: sudo port install ffmpeg"
            echo "   • Manual: https://ffmpeg.org"
            echo ""
            echo "   Continuing anyway - the app may still work with direct numpy processing..."
        fi
    fi
    
    echo "✅ Installation completed!"
    echo ""
else
    echo "✅ WhisperBar is already installed"
    echo ""
fi

# Activate virtual environment (in case it wasn't activated above)
source venv/bin/activate

# Check permissions
echo "🔐 Checking permissions..."

# Check if accessibility permissions are granted
echo "🔐 Checking accessibility permissions..."
python3 -c "
import sys
try:
    from pynput import keyboard
    def dummy(key): pass
    listener = keyboard.Listener(on_press=dummy)
    listener.start()
    listener.stop()
    print('✅ Accessibility permissions appear to be granted')
except Exception as e:
    print('⚠️  Accessibility permissions may be needed')
    print(f'   Error: {e}')
    print('')
    print('📋 If pasting doesn\'t work, follow these steps:')
    print('1. Go to System Settings > Privacy & Security > Accessibility')
    print('2. Click the \"+\" button')
    print('3. Find and add:')
    print('   • Ghostty (if using Ghostty from Raycast)')  
    print('   • Terminal (if using Terminal)')
    print('   • iTerm (if using iTerm)')
    print('   • Python (or Python Launcher)')
    print('   • osascript (for AppleScript paste method)')
    print('4. Make sure all are checked/enabled')
    print('5. IMPORTANT: If using Raycast → Ghostty, you must add Ghostty!')
    print('')
    print('📝 Note: The app will still work with clipboard-only mode')
    print('')
" 2>/dev/null

echo ""
echo "🎤 Starting WhisperBar..."
echo "📝 Instructions:"
echo "   • Hold RIGHT SHIFT to record"
echo "   • Release RIGHT SHIFT to transcribe and paste"
echo "   • Click the 🎤 icon in menu bar for settings"
echo "   • Press Ctrl+C here to quit"
echo ""
echo "🔄 Loading model (may take a moment)..."

# Launch the application
python main.py 