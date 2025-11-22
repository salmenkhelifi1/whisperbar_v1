#!/bin/bash

# WhisperBar Installer Script
# This script installs WhisperBar and all dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     WhisperBar Installation Script    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}❌ Error: This app only works on macOS${NC}"
    exit 1
fi

# Check Python version
echo -e "${YELLOW}🔍 Checking Python version...${NC}"
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    echo -e "${RED}❌ Error: Python 3.9+ is required${NC}"
    echo -e "${YELLOW}   Please install Python 3.9+ from https://python.org${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✅ Found: $PYTHON_VERSION${NC}"
echo ""

# Check for Homebrew
echo -e "${YELLOW}🔍 Checking for Homebrew...${NC}"
if command -v brew &> /dev/null; then
    echo -e "${GREEN}✅ Homebrew found${NC}"
    BREW_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  Homebrew not found (optional, but recommended)${NC}"
    BREW_AVAILABLE=false
fi
echo ""

# Check/create virtual environment
echo -e "${YELLOW}📦 Setting up virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}   Virtual environment exists, removing old one...${NC}"
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment created${NC}"
echo ""

# Upgrade pip
echo -e "${YELLOW}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✅ pip upgraded${NC}"
echo ""

# Install Python dependencies
echo -e "${YELLOW}📥 Installing Python dependencies...${NC}"
echo -e "${BLUE}   This may take 5-10 minutes (downloading large packages)...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✅ Python dependencies installed${NC}"
echo ""

# Check/Install ffmpeg
echo -e "${YELLOW}🔧 Checking for ffmpeg...${NC}"
FFMPEG_FOUND=false
for path in "/opt/homebrew/bin/ffmpeg" "/usr/local/bin/ffmpeg" "$(which ffmpeg 2>/dev/null)"; do
    if [ -x "$path" ]; then
        echo -e "${GREEN}✅ ffmpeg found at: $path${NC}"
        FFMPEG_FOUND=true
        break
    fi
done

if [ "$FFMPEG_FOUND" = false ]; then
    echo -e "${YELLOW}📦 ffmpeg not found, installing...${NC}"
    if [ "$BREW_AVAILABLE" = true ]; then
        brew install ffmpeg
        echo -e "${GREEN}✅ ffmpeg installed${NC}"
    else
        echo -e "${RED}⚠️  Please install ffmpeg manually:${NC}"
        echo -e "${YELLOW}   1. Install Homebrew: https://brew.sh${NC}"
        echo -e "${YELLOW}   2. Run: brew install ffmpeg${NC}"
        echo -e "${YELLOW}   Or download from: https://ffmpeg.org${NC}"
    fi
fi
echo ""

# Create launcher script
echo -e "${YELLOW}📝 Creating launcher script...${NC}"
cat > run_whisperbar.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python main.py
EOF
chmod +x run_whisperbar.sh
echo -e "${GREEN}✅ Launcher script created${NC}"
echo ""

# Check permissions
echo -e "${YELLOW}🔐 Checking permissions...${NC}"
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
    print('')
    print('📋 To enable full functionality:')
    print('1. Go to System Settings > Privacy & Security > Accessibility')
    print('2. Click the \"+\" button')
    print('3. Add Terminal (or your terminal app)')
    print('4. Restart WhisperBar')
" 2>/dev/null || echo -e "${YELLOW}⚠️  Could not check permissions${NC}"
echo ""

# Installation complete
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   ✅ Installation Complete!            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📋 To run WhisperBar:${NC}"
echo -e "${YELLOW}   ./run_whisperbar.sh${NC}"
echo -e "${YELLOW}   or${NC}"
echo -e "${YELLOW}   ./QuickStart.sh${NC}"
echo ""
echo -e "${BLUE}📋 Available Models:${NC}"
echo -e "   • Tiny (.en) - Fastest, English only"
echo -e "   • Small (.en) - Fast, English only"
echo -e "   • Medium (.en) - Balanced (default)"
echo -e "   • Base (.en) - OpenAI base model"
echo -e "   • Tiny (Multilingual) - Fastest, 99+ languages"
echo -e "   • Base (Multilingual) - Base multilingual"
echo -e "   • Small (Multilingual) - Small multilingual"
echo -e "   • Medium (Multilingual) - Medium multilingual"
echo -e "   • Large (v2) - High accuracy multilingual"
echo -e "   • Large (v3.5) - Highest accuracy multilingual"
echo ""
echo -e "${BLUE}🎤 Enjoy WhisperBar!${NC}"

