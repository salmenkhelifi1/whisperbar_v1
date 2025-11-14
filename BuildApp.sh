#!/bin/bash

# WhisperBar App Builder
# Creates a standalone .app bundle

set -e

echo "📱 WhisperBar App Builder"
echo "========================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run QuickStart.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Install PyInstaller if not already installed
echo "📦 Installing PyInstaller..."
pip install pyinstaller

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Build the app
echo "🔨 Building WhisperBar.app..."
echo "   This may take several minutes..."
pyinstaller speechtotext.spec

# Check if build was successful
if [ -d "dist/WhisperBar.app" ]; then
    echo ""
    echo "🎉 Success! WhisperBar.app has been built."
    echo ""
    echo "📍 Location: $(pwd)/dist/WhisperBar.app"
    echo ""
    echo "📋 Next steps:"
    echo "1. Copy WhisperBar.app to your Applications folder:"
    echo "   cp -r dist/WhisperBar.app /Applications/"
    echo ""
    echo "2. Or launch it directly:"
    echo "   open dist/WhisperBar.app"
    echo ""
    echo "3. Grant accessibility permissions when prompted"
    echo ""
    
    # Ask if user wants to copy to Applications
    read -p "❓ Copy to Applications folder now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📁 Copying to Applications..."
        cp -r dist/WhisperBar.app /Applications/
        echo "✅ Done! You can now find WhisperBar in your Applications folder."
        echo "🚀 Launch it from Spotlight (⌘+Space, type 'WhisperBar')"
    fi
    
    # Ask if user wants to launch it now
    read -p "❓ Launch WhisperBar now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Launching WhisperBar..."
        open dist/WhisperBar.app
    fi
    
else
    echo "❌ Build failed. Check the output above for errors."
    exit 1
fi 