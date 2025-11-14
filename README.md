# WhisperBar - Intelligent Speech-to-Text Status Bar App

A macOS status bar application that provides intelligent speech-to-text transcription using OpenAI's Whisper models. Features three optimized processing modes and push-to-talk functionality with automatic text pasting.

## ✨ Features

- 🎤 **Push-to-Talk**: Hold Right Shift to record, release to transcribe
- 🔄 **Auto-Paste**: Automatically pastes transcription where your cursor is
- 🧠 **Three Processing Modes**: Traditional, Optimized, and Ultra-Fast
- ⚡ **Apple Silicon Optimized**: Uses MPS acceleration on Apple Silicon Macs
- 🎯 **Smart VAD**: Voice Activity Detection for better speech segmentation
- 📋 **Clipboard Integration**: Copies to clipboard as backup
- 🔔 **Rich Status Updates**: Real-time feedback with timing and performance stats
- 🌍 **Multilingual Support**: Automatic language detection and transcription

## 🚀 Processing Modes

### 🎯 **Optimized** (Default)
- Smart segmentation + your selected model
- Best balance of speed and accuracy
- Recommended for most users

### 🧠 **Traditional** 
- Proven VAD processing
- Most reliable for challenging audio
- Baseline processing method

### ⚡ **Ultra-Fast**
- Forces Tiny model + maximum optimization
- 3-5x real-time speed
- Perfect for quick notes

## 📋 Requirements

- **macOS 10.15+** (Catalina or later)
- **Python 3.9+**
- **Apple Silicon Mac** (recommended) or Intel Mac
- **~2GB free disk space** (for model storage)

## 🛠 Installation

### Quick Start
```bash
# Clone and setup
git clone https://github.com/yourusername/WhisperBar.git
cd WhisperBar

# Run the setup script
chmod +x QuickStart.sh
./QuickStart.sh
```

### Manual Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Grant permissions and run
python main.py
```

### 🔐 Required Permissions

1. **Accessibility**: Go to **System Settings** > **Privacy & Security** > **Accessibility**
2. Click **"+"** and add your **Terminal** (or Python app)
3. **Microphone**: Grant microphone access when prompted

## 🎮 Usage

1. **Start**: Run `python main.py` or double-click `WhisperBar.command`
2. **Record**: Hold **Right Shift** key and speak
3. **Transcribe**: Release key - text appears instantly where your cursor is
4. **Configure**: Click status bar icon for settings

### Status Bar Menu
- **Select Model**: Choose from Tiny, Small, Medium, Large
- **Processing Mode**: Switch between Traditional, Optimized, Ultra-Fast  
- **Reload Model**: Refresh current model
- **Quit**: Exit application

## 🤖 Model Options

| Model | Size | Speed | Languages | Best For |
|-------|------|-------|-----------|----------|
| **Tiny (.en)** | ~39MB | ⚡⚡⚡ | English | Ultra-fast notes |
| **Small (.en)** | ~244MB | ⚡⚡ | English | Quick transcription |
| **Medium (.en)** | ~769MB | ⚡ | English | Balanced (default) |
| **Large (v3.5)** | ~1.5GB | 🐌 | 99+ languages | Maximum accuracy |

## ⚙️ Configuration

Key settings in `config.py`:
```python
# Processing mode: "traditional", "optimized", "ultra_fast"
PROCESSING_MODE = "optimized"

# Default model
DEFAULT_MODEL_NAME = "Medium (.en)"

# VAD sensitivity (0.0-1.0)
VAD_THRESHOLD = 0.3
```

## 🔧 Troubleshooting

### 🔑 Key Detection Issues
- Verify Accessibility permissions are granted
- Restart the application
- Check no other apps intercept Right Shift

### 🎤 Audio Problems  
- Grant microphone permissions
- Test microphone in other apps
- Check audio input levels

### 🤖 Model Issues
- Ensure stable internet (first download)
- Verify 2GB+ free disk space
- Try reloading model from menu

### 📝 Text Not Pasting
- Grant Accessibility permissions
- Test in TextEdit first
- Check cursor is in a text field

## 🏗 Building Standalone App

Create a `.app` bundle for easy distribution:
```bash
# Install PyInstaller
pip install pyinstaller

# Build app
./BuildApp.sh

# Find app in dist/ folder
```

## 🎯 Performance Tips

- **Default Setup**: Optimized mode with Medium model works great
- **Speed Priority**: Use Ultra-Fast mode for quick notes
- **Accuracy Priority**: Use Traditional mode with Large model
- **First Run**: Initial model download takes time (then cached)
- **Memory**: Large model uses ~2GB RAM when loaded

## 📊 Performance Benchmarks

| Mode | Model | Speed | Typical Time |
|------|-------|-------|--------------|
| Ultra-Fast | Tiny | 5x real-time | 0.5-0.8s |
| Optimized | Medium | 3x real-time | 0.8-2.0s |
| Traditional | Large | 1x real-time | 2.0-5.0s |

## 🔮 Advanced Features

- **VAD Optimization**: Smart speech detection with configurable thresholds
- **MPS Acceleration**: Apple Silicon GPU acceleration with float16 optimization
- **Thread Safety**: Non-blocking processing with proper error handling
- **Memory Management**: Efficient model loading and caching
- **AppleScript Integration**: Seamless text pasting across all apps

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **OpenAI** for the Whisper models
- **Hugging Face** for the transformers library
- **PyTorch** for the ML framework
- **pynput** for keyboard/mouse handling 