# How WhisperBar Works

## 🎯 Overview

**WhisperBar** is a macOS menu bar application that converts speech to text in real-time. It runs **100% locally** on your Mac - no internet required after initial setup.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    macOS Menu Bar (rumps)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Status Icon │  │  Model Menu  │  │  Mode Menu   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Keyboard Listener (pynput)                       │
│  • Listens for Right Option key (push-to-talk)              │
│  • Global hotkey detection                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Audio Recording (sounddevice)                    │
│  • Captures microphone input at 16kHz                        │
│  • Stores audio frames in memory                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Transcription Engine (Faster Whisper / Whisper)      │
│  • Processes audio → text                                    │
│  • Uses AI models (loaded locally)                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Text Output (pynput + pyperclip)                 │
│  • Copies to clipboard                                       │
│  • Auto-pastes where cursor is                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete Flow: From Speech to Text

### 1. **Startup Phase** (When you run `python main.py`)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. App Initialization                                        │
│    • Creates menu bar icon (🎤 Mic)                          │
│    • Sets up queues for thread communication                 │
│    • Loads default model (Large Multilingual)                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Model Loading (LOCAL - No Internet Needed)               │
│    • Downloads model from Hugging Face (first time only)      │
│    • Caches model in ~/.cache/huggingface/                   │
│    • Loads into memory (RAM)                                 │
│    • Uses Faster Whisper (2-4x faster) or regular Whisper    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Background Services Start                                  │
│    • Keyboard listener starts (watches for Right Option key) │
│    • VAD model loads (for speech detection)                  │
│    • Status updates begin                                    │
└─────────────────────────────────────────────────────────────┘
```

### 2. **Recording Phase** (Hold Right Option Key)

```
User holds Right Option key
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Keyboard Listener Detects Key Press                           │
│    • on_press() callback fires                                │
│    • Cancels any ongoing transcription                        │
│    • Resets state                                             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Start Recording                                               │
│    • Checks if model is loaded                                │
│    • Opens audio stream (sounddevice)                         │
│    • Sets is_recording = True                                 │
│    • Menu icon changes to "🔴 Rec"                             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Audio Capture Loop (Background Thread)                        │
│    • audio_callback() called continuously                     │
│    • Captures audio chunks (frames)                           │
│    • Stores in audio_frames[] array                          │
│    • Continues until key released                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. **Transcription Phase** (Release Right Option Key)

```
User releases Right Option key
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Stop Recording                                                │
│    • on_release() callback fires                              │
│    • Closes audio stream                                      │
│    • Concatenates all audio frames                            │
│    • Converts to numpy array (float32)                       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Start Transcription Thread                                    │
│    • Creates background thread                                │
│    • Menu icon changes to "⚡ Trans"                          │
│    • Processes audio → text                                  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Transcription Processing (Choose Method)                     │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Method 1: Faster Whisper (Default, 2-4x faster)     │    │
│  │   • Uses CTranslate2 backend                         │    │
│  │   • Optimized C++ implementation                     │    │
│  │   • Processes entire audio                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Method 2: Regular Whisper (Fallback)                 │    │
│  │   • Uses Hugging Face transformers                   │    │
│  │   • Three processing modes:                          │    │
│  │     - Traditional: VAD segmentation                   │    │
│  │     - Optimized: Smart segmentation                  │    │
│  │     - Ultra-Fast: No VAD, direct processing          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Get Transcription Result                                      │
│    • Text extracted from model output                         │
│    • Cleaned and formatted                                    │
│    • Put into transcription_queue                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. **Output Phase** (Text Appears)

```
Transcription ready
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Copy to Clipboard                                            │
│    • Always copies text to clipboard first                   │
│    • Uses pyperclip library                                  │
│    • Ensures text is available even if paste fails           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Auto-Paste Attempt                                           │
│    • Method 1: pynput (Cmd+V simulation)                     │
│    • Method 2: AppleScript (fallback)                        │
│    • Pastes where cursor is                                  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Show Notification                                            │
│    • Success: "✅ Text Pasted!"                               │
│    • Fallback: "📋 Text Ready - Press Cmd+V"                │
│    • Menu icon: "✅ Ready"                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Key Components Explained

### 1. **Model Loading System**

**Models are 100% LOCAL** - downloaded once, cached forever:

```python
# First run: Downloads from Hugging Face
# Location: ~/.cache/huggingface/hub/models--openai--whisper-large-v3/

# Subsequent runs: Loads from local cache (instant)
# No internet connection needed!
```

**Two Engine Options:**

- **Faster Whisper** (default): Uses CTranslate2, 2-4x faster
- **Regular Whisper**: Uses Hugging Face transformers, more compatible

**Model Caching:**

- Models cached in memory for instant switching
- Multiple models can be preloaded
- Saves time when switching between models

### 2. **Threading Architecture**

The app uses **multiple threads** for non-blocking operation:

```
Main Thread (rumps)
  ├── UI updates (menu bar)
  ├── Queue checker (status/transcription)
  └── Menu interactions

Background Threads:
  ├── Keyboard listener thread (watches for key presses)
  ├── Audio capture thread (records microphone)
  ├── Transcription thread (processes audio → text)
  └── Model loading thread (loads models in background)
```

**Thread-Safe Communication:**

- `status_queue`: Status updates (loading, ready, error)
- `transcription_queue`: Completed transcriptions
- Global flags: `is_recording`, `is_processing`

### 3. **Audio Processing Pipeline**

```
Microphone Input (16kHz, mono)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Audio Normalization                                           │
│    • Prevents clipping                                        │
│    • Ensures optimal volume                                  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Voice Activity Detection (VAD) - Optional                   │
│    • Detects speech segments                                 │
│    • Skips silence                                           │
│    • Speeds up processing                                    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Smart Segmentation (Optimized Mode)                           │
│    • Splits long audio into chunks                           │
│    • Processes chunks optimally                              │
│    • Merges results                                          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Model Inference                                               │
│    • Whisper model processes audio                            │
│    • Generates text tokens                                   │
│    • Decodes to readable text                                 │
└─────────────────────────────────────────────────────────────┘
```

### 4. **Processing Modes**

**Traditional Mode:**

- Uses VAD to detect speech segments
- Processes each segment separately
- Most reliable, baseline method

**Optimized Mode (Default):**

- Smart segmentation + VAD
- Balances speed and accuracy
- Uses your selected model

**Ultra-Fast Mode:**

- Skips VAD entirely
- Forces Tiny model
- Maximum speed (3-5x real-time)

---

## 🔐 Permissions & Security

**macOS Permissions Required:**

1. **Accessibility** (for global key detection)
   - System Settings → Privacy & Security → Accessibility
   - Allows app to detect Right Option key globally

2. **Microphone** (for audio recording)
   - Prompted automatically on first run
   - Required for speech capture

**Privacy:**

- ✅ All processing happens **locally** on your Mac
- ✅ No data sent to external servers
- ✅ Models cached locally
- ✅ Audio never leaves your computer

---

## 💾 Storage & Memory

**Disk Space:**

- Models: ~1-2GB (downloaded once, cached)
- Location: `~/.cache/huggingface/`

**Memory Usage:**

- Large model: ~2GB RAM when loaded
- Medium model: ~800MB RAM
- Tiny model: ~100MB RAM

**Caching:**

- Models cached in memory for instant switching
- Audio buffers pre-allocated for speed

---

## 🎮 User Interaction Flow

### Method 1: Keyboard (Push-to-Talk)

```
1. Hold Right Option key → Recording starts
2. Speak into microphone
3. Release Right Option key → Transcription starts
4. Text appears where cursor is
```

### Method 2: Menu Click

```
1. Click menu bar icon
2. Click "🎤 Start Recording"
3. Speak
4. Click "⏹ Stop Recording"
5. Text appears where cursor is
```

### Menu Options

- **Select Model**: Choose accuracy vs speed
- **Processing Mode**: Traditional/Optimized/Ultra-Fast
- **⚡ Faster Whisper**: Toggle Faster Whisper on/off
- **Reload Model**: Refresh current model

---

## 🚀 Performance Optimizations

1. **Faster Whisper**: 2-4x faster than regular Whisper
2. **Model Caching**: Instant model switching
3. **MPS Acceleration**: Uses Apple Silicon GPU (if available)
4. **Float16**: Faster inference with minimal accuracy loss
5. **VAD**: Skips silence, processes only speech
6. **Smart Segmentation**: Optimizes chunk sizes
7. **Pre-allocated Buffers**: Reduces memory allocation overhead
8. **Threading**: Non-blocking UI, parallel processing

---

## 🔧 Troubleshooting

**Model Not Loading:**

- Check internet connection (first download only)
- Verify disk space (~2GB free)
- Check logs for errors

**No Transcription:**

- Verify model is loaded (check menu status)
- Check microphone permissions
- Ensure audio is being captured

**Key Not Working:**

- Verify Accessibility permissions
- Check no other app intercepts Right Option key
- Try menu click method instead

**Text Not Pasting:**

- Check Accessibility permissions
- Try manual paste (Cmd+V)
- Text is always in clipboard as backup

---

## 🌐 Cloud API Support

**WhisperBar now supports cloud APIs as an alternative to local models:**

### Supported Cloud Providers

- **OpenAI Whisper API**: High-quality cloud transcription
- **Google Speech-to-Text**: Google's cloud transcription service
- **Deepgram**: Fast, accurate cloud transcription
- **Custom API**: Use your own transcription endpoint

### Configuration

1. Set `USE_CLOUD_API = True` in `config.py`
2. Choose provider: `CLOUD_PROVIDER = "openai"` (or "google", "deepgram", "custom")
3. Set API keys via environment variables:
   - `OPENAI_API_KEY` for OpenAI
   - `GOOGLE_API_KEY` for Google
   - `DEEPGRAM_API_KEY` for Deepgram
   - `CUSTOM_API_URL` for custom APIs

### Benefits

- ✅ No local model download needed
- ✅ Always uses latest cloud models
- ✅ Automatic fallback to local if cloud fails
- ✅ Lower memory usage (no local model loaded)

## ⌨️ Customizable Keyboard Shortcuts

**All keyboard shortcuts are now customizable via `config.py`:**

- `TRIGGER_KEY`: Push-to-talk key (default: "alt_r")
- `TOGGLE_RECORDING_KEY`: Toggle recording (default: "cmd+shift+w")
- `QUIT_APP_KEY`: Quit app (default: "cmd+shift+q")
- `RELOAD_MODEL_KEY`: Reload model (default: "cmd+shift+r")
- `TOGGLE_APP_KEY`: Enable/disable app (default: "cmd+shift+t")

**Format:** Use "alt_r" for single keys, "cmd+shift+w" for combinations.

## 🎯 Auto-Model Selection

**WhisperBar automatically selects the best model for your Mac:**

- **Apple Silicon + 16+ GB RAM**: Large (Multilingual) - Best accuracy
- **Apple Silicon + 8-16 GB RAM**: Medium (.en) - Balanced
- **Apple Silicon + <8 GB RAM**: Tiny (.en) - Fast, low memory
- **Intel Mac + 8+ GB RAM**: Medium (.en) - Balanced
- **Intel Mac + <8 GB RAM**: Tiny (.en) - Fast, low memory

Select "🎯 Auto-Select Best" from the model menu to re-run auto-detection.

## 🔋 Memory & Battery Optimizations

**New optimization features:**

- **Lazy Model Loading**: Models load only when first recording starts
- **Auto-Unload Models**: Models unload when not in use (saves memory)
- **Low Power Mode**: Reduces CPU usage when idle
- **Disabled Background Preloading**: Saves battery by not preloading models
- **Reduced Buffer Sizes**: Smaller audio buffers in low power mode

Configure in `config.py`:

- `LAZY_MODEL_LOADING = True`
- `AUTO_UNLOAD_MODELS = True`
- `ENABLE_LOW_POWER_MODE = False`
- `DISABLE_BACKGROUND_PRELOADING = True`

## 📊 Summary

**WhisperBar is a flexible speech-to-text app that:**

- ✅ Runs locally OR uses cloud APIs (your choice)
- ✅ Auto-selects best model for your Mac
- ✅ Customizable keyboard shortcuts
- ✅ Memory and battery optimized
- ✅ Processes audio → text in real-time
- ✅ Auto-pastes results where you're typing
- ✅ Works with keyboard shortcut or menu
- ✅ Supports multiple models and processing modes
- ✅ Optimized for speed and accuracy

**Key Point:** Choose local (offline) or cloud (online) - both work seamlessly!
