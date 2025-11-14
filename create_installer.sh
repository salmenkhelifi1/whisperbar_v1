#!/bin/bash

# Create WhisperBar Installer Package

echo "📦 Creating WhisperBar Installer Package"
echo "========================================"

# Create installer directory
INSTALLER_DIR="WhisperBar_Installer"
rm -rf "$INSTALLER_DIR"
mkdir "$INSTALLER_DIR"

# Copy necessary files
echo "📁 Copying files..."
cp main.py "$INSTALLER_DIR/"
cp config.py "$INSTALLER_DIR/"
cp utils.py "$INSTALLER_DIR/"
cp requirements.txt "$INSTALLER_DIR/"
cp README.md "$INSTALLER_DIR/"
cp QuickStart.sh "$INSTALLER_DIR/"
cp BuildApp.sh "$INSTALLER_DIR/"
cp WhisperBar.command "$INSTALLER_DIR/"
cp speechtotext.spec "$INSTALLER_DIR/"

# Create a simple installer script
cat > "$INSTALLER_DIR/INSTALL.command" << 'EOF'
#!/bin/bash

# WhisperBar Installer

clear
echo "🎤 WhisperBar Installer"
echo "======================"
echo ""
echo "Welcome to WhisperBar - Speech-to-Text for macOS!"
echo ""
echo "This installer will:"
echo "• Install Python dependencies"
echo "• Set up the virtual environment" 
echo "• Guide you through permission setup"
echo ""
read -p "Press Enter to continue..."

# Get the directory where this installer is located
INSTALLER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$INSTALLER_DIR"

# Run the QuickStart script
exec ./QuickStart.sh
EOF

chmod +x "$INSTALLER_DIR/INSTALL.command"

# Create a README for the installer
cat > "$INSTALLER_DIR/README_INSTALLER.txt" << 'EOF'
🎤 WhisperBar - Speech-to-Text App for macOS

QUICK START:
============
1. Double-click "INSTALL.command" to install and run
2. Grant accessibility permissions when prompted
3. Use Right Shift to record speech!

ALTERNATIVE OPTIONS:
===================
• Double-click "WhisperBar.command" for quick launch
• Run "./QuickStart.sh" in Terminal
• Run "./BuildApp.sh" to create standalone .app

WHAT IT DOES:
=============
• Hold Right Shift = Record speech
• Release Right Shift = Transcribe and paste text
• Works in any app where you can type

REQUIREMENTS:
=============
• macOS 10.15+
• Python 3.9+
• Microphone access
• Accessibility permissions

For detailed instructions, see README.md
EOF

# Create a simple uninstaller
cat > "$INSTALLER_DIR/UNINSTALL.command" << 'EOF'
#!/bin/bash

echo "🗑️  WhisperBar Uninstaller"
echo "========================="
echo ""
echo "This will remove:"
echo "• Virtual environment and dependencies"
echo "• Cache files"
echo "• WhisperBar.app (if installed)"
echo ""
read -p "Are you sure you want to uninstall? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Get the directory where this script is located
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    cd "$SCRIPT_DIR"
    
    echo "🧹 Removing files..."
    rm -rf venv/
    rm -rf ~/.whisperbar_cache/
    rm -rf build/ dist/
    rm -f temp_recording.wav
    rm -rf /Applications/WhisperBar.app
    
    echo "✅ WhisperBar has been uninstalled."
    echo "Note: You may need to manually remove accessibility permissions"
    echo "from System Settings > Privacy & Security > Accessibility"
else
    echo "❌ Uninstall cancelled."
fi

read -p "Press Enter to close..."
EOF

chmod +x "$INSTALLER_DIR/UNINSTALL.command"

echo ""
echo "✅ Installer package created: $INSTALLER_DIR/"
echo ""
echo "📋 Contents:"
ls -la "$INSTALLER_DIR/"
echo ""
echo "🎯 To distribute:"
echo "1. Zip the '$INSTALLER_DIR' folder"
echo "2. Users just need to:"
echo "   • Unzip"
echo "   • Double-click INSTALL.command"
echo "   • Follow the prompts"
echo ""

# Ask if user wants to create a zip file
read -p "❓ Create a zip file for distribution? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ZIP_NAME="WhisperBar_Installer.zip"
    echo "📦 Creating $ZIP_NAME..."
    zip -r "$ZIP_NAME" "$INSTALLER_DIR/" -x "*.DS_Store"
    echo "✅ Created: $ZIP_NAME"
    echo "📤 This file can be shared for easy installation!"
fi 