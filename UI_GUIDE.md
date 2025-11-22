# WhisperBar UI Guide

## 🎯 Quick Start

1. **Launch the app**: Run `python main.py` from terminal
2. **Look for the menu bar icon**: You'll see "🎤 Mic" in your macOS menu bar (top right)
3. **Click the icon** to open the menu

---

## 📋 Menu Bar Interface

### Main Menu Items

When you click the menu bar icon, you'll see:

```
🎤 Start Recording (Default)
─────────────────────────────
Keyboard: Hold Option Key
─────────────────────────────
Select Model
Processing Mode
⚡ Faster Whisper
Keyboard Shortcuts
─────────────────────────────
Quit
```

---

## 🎤 Recording Methods

### Method 1: Keyboard Shortcut (Push-to-Talk) - RECOMMENDED

1. **Hold** the **Right Option** key (or your customized trigger key)
2. **Speak** into your microphone
3. **Release** the key when done
4. **Text appears** automatically where your cursor is!

**Note**: The app will show "🔴 Rec" while recording.

### Method 2: Menu Click

1. Click the menu bar icon
2. Click **"🎤 Start Recording (Default)"**
3. Speak into your microphone
4. Click **"⏹ Stop Recording"** when done
5. Text appears automatically!

**Auto-stop**: Recording automatically stops after 15 seconds for safety.

---

## 🤖 Model Selection

Click **"Select Model"** to see available models:

### 🎯 Auto-Select Best

- Automatically picks the best model for your Mac
- Considers: Apple Silicon vs Intel, RAM, GPU availability
- Click this first to get optimal performance!

### 🏠 Local Models

**Large (Multilingual)** 🐌 🎯🎯🎯 (1.5GB)

- Best accuracy, handles accents well
- Best for: High accuracy needs, non-native speakers
- Requires: 16+ GB RAM on Apple Silicon

**Medium (.en)** ⚡⚡ 🎯🎯 (769MB)

- Balanced speed and accuracy
- Best for: General use, English speakers
- Recommended for most users

**Tiny (.en)** ⚡⚡⚡ 🎯 (39MB)

- Fastest, lowest memory
- Best for: Quick notes, low-end Macs
- Trade-off: Lower accuracy

### ☁️ Cloud APIs (if enabled)

If you've enabled cloud APIs in `config.py`:

- **☁️ OpenAI** - High-quality cloud transcription
- **☁️ Google** - Google's cloud service
- **☁️ Deepgram** - Fast cloud transcription
- **☁️ Custom** - Your own API endpoint

**Note**: Cloud APIs require internet and API keys.

---

## ⚙️ Processing Modes

Click **"Processing Mode"** to choose:

### 🎯 Optimized (Default)

- Smart segmentation + your selected model
- Best balance of speed and accuracy
- **Recommended for most users**

### 📊 Traditional

- Proven VAD processing
- Most reliable for challenging audio
- Best for: Difficult audio conditions

### ⚡ Ultra-Fast

- Forces Tiny model + maximum optimization
- 3-5x real-time speed
- Best for: Quick notes, speed priority

---

## ⚡ Faster Whisper Toggle

**⚡ Faster Whisper** checkbox

- ✅ **Checked**: Uses Faster Whisper (2-4x faster)
- ☐ **Unchecked**: Uses regular Whisper (more compatible)

**Recommendation**: Keep it checked for best performance!

---

## ⌨️ Keyboard Shortcuts

Click **"Keyboard Shortcuts"** to see all shortcuts:

### Default Shortcuts

- **Push-to-Talk**: `Right Option` key (hold to record)
- **Toggle Recording**: `Cmd+Shift+W` (toggle on/off without holding)
- **Reload Model**: `Cmd+Shift+R` (refresh current model)
- **Toggle App**: `Cmd+Shift+T` (enable/disable app)
- **Quit App**: `Cmd+Shift+Q` (exit application)

### Customizing Shortcuts

Edit `config.py` to change shortcuts:

```python
TRIGGER_KEY = "alt_r"  # Change to "f1", "space", etc.
TOGGLE_RECORDING_KEY = "cmd+shift+w"  # Change combination
```

**Format**:

- Single key: `"alt_r"`, `"f1"`, `"space"`
- Combination: `"cmd+shift+w"`, `"ctrl+alt+r"`

---

## 📊 Status Bar Indicators

The menu bar icon shows different states:

- **🎤 Mic** - Idle, ready to record
- **🔴 Rec** - Currently recording
- **⚡ Trans** - Transcribing audio
- **🚀 Load** - Loading model
- **✅ Ready** - Model loaded, ready to use
- **❌ Error** - Something went wrong
- **🧠 Proc** - Processing audio
- **🎯 Opt** - Optimizing

**Status updates** show:

- Model name
- Processing mode
- Character count
- Processing time
- Cloud API indicator (☁️) if using cloud

---

## 💡 Tips & Tricks

### 1. **First Time Setup**

- Click "🎯 Auto-Select Best" to get optimal model
- Grant microphone permissions when prompted
- Grant Accessibility permissions for keyboard shortcuts

### 2. **Best Performance**

- Use **Medium (.en)** model for balanced performance
- Keep **⚡ Faster Whisper** enabled
- Use **Optimized** processing mode

### 3. **Battery Saving**

- Enable `ENABLE_LOW_POWER_MODE = True` in `config.py`
- Enable `LAZY_MODEL_LOADING = True` (loads model only when needed)
- Enable `DISABLE_BACKGROUND_PRELOADING = True`

### 4. **Memory Saving**

- Enable `AUTO_UNLOAD_MODELS = True` in `config.py`
- Use smaller models (Tiny or Medium) if low on RAM
- Use cloud APIs to avoid loading local models

### 5. **Cloud API Setup**

1. Set `USE_CLOUD_API = True` in `config.py`
2. Choose provider: `CLOUD_PROVIDER = "openai"`
3. Set API key: `export OPENAI_API_KEY="your-key"`
4. Restart app

### 6. **Troubleshooting**

**No text appearing?**

- Check if model is loaded (should show "✅ Ready")
- Verify microphone permissions
- Try clicking "Reload Current Model"

**Keyboard shortcut not working?**

- Check Accessibility permissions
- Try menu click method instead
- Verify shortcut in "Keyboard Shortcuts" menu

**App crashed?**

- Check logs in terminal
- Try reloading model
- Restart the app

---

## 🎮 Common Workflows

### Quick Note Taking

1. Hold Right Option key
2. Speak quickly
3. Release key
4. Text appears instantly!

### Long Transcription

1. Click menu → Start Recording
2. Speak for up to 15 seconds
3. Click Stop Recording
4. Wait for transcription

### Switch Models

1. Click "Select Model"
2. Choose desired model
3. Wait for "✅ Ready" status
4. Start recording!

### Use Cloud API

1. Set up API key in environment
2. Edit `config.py`: `USE_CLOUD_API = True`
3. Select cloud provider from menu
4. Start recording (no local model needed!)

---

## 📱 Menu Structure Reference

```
🎤 Mic (Menu Bar Icon)
│
├── 🎤 Start Recording / ⏹ Stop Recording
│
├── ─────────────────────────────
│
├── Keyboard: Hold Option Key (info)
│
├── ─────────────────────────────
│
├── 📋 Select Model
│   ├── 🎯 Auto-Select Best
│   ├── ─────────────────────────────
│   ├── 🏠 Local Models
│   │   ├── Large (Multilingual) 🐌 🎯🎯🎯 (1.5GB) ✓
│   │   ├── Medium (.en) ⚡⚡ 🎯🎯 (769MB)
│   │   └── Tiny (.en) ⚡⚡⚡ 🎯 (39MB)
│   └── ☁️ Cloud APIs (if enabled)
│       ├── ☁️ OpenAI
│       ├── ☁️ Google
│       ├── ☁️ Deepgram
│       └── ☁️ Custom
│
├── ⚙️ Processing Mode
│   ├── Traditional Processing
│   ├── Optimized Processing ✓
│   └── Ultra-Fast Processing
│
├── ⚡ Faster Whisper ✓
│
├── ⌨️ Keyboard Shortcuts
│   ├── Push-to-Talk: alt_r
│   ├── Toggle Recording: cmd+shift+w
│   ├── Reload Model: cmd+shift+r
│   ├── Toggle App: cmd+shift+t
│   └── Quit App: cmd+shift+q
│
└── Quit
```

---

## 🚀 Getting Started Checklist

- [ ] Run `python main.py`
- [ ] Grant microphone permissions
- [ ] Grant Accessibility permissions (for keyboard shortcuts)
- [ ] Click menu → "🎯 Auto-Select Best" model
- [ ] Wait for "✅ Ready" status
- [ ] Try recording: Hold Right Option key and speak
- [ ] Check if text appears where cursor is
- [ ] Customize shortcuts in `config.py` if needed
- [ ] Set up cloud API (optional) if preferred

---

## ❓ FAQ

**Q: How do I know which model to use?**
A: Click "🎯 Auto-Select Best" - it picks the optimal model for your Mac!

**Q: Can I use cloud APIs without downloading models?**
A: Yes! Set `USE_CLOUD_API = True` and configure your API key.

**Q: Why isn't my keyboard shortcut working?**
A: Check Accessibility permissions in System Settings → Privacy & Security → Accessibility.

**Q: How do I change the trigger key?**
A: Edit `TRIGGER_KEY` in `config.py` (e.g., `"f1"`, `"space"`, `"alt_l"`).

**Q: Can I disable the app without quitting?**
A: Yes! Press `Cmd+Shift+T` to toggle app on/off.

**Q: How do I see what shortcuts are configured?**
A: Click menu → "Keyboard Shortcuts" to see all current shortcuts.

---

Enjoy using WhisperBar! 🎉
