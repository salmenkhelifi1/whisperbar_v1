# Configuration file for WhisperBar
import os
from pynput import keyboard

# --- App Configuration ---
APP_NAME = "WhisperBar"
APP_VERSION = "1.0.0"

# --- Status Bar Icons ---
APP_ICON_IDLE = "🎤"
APP_ICON_RECORDING = "🔴"
APP_ICON_TRANSCRIBING = "⚡"
APP_ICON_LOADING = "🚀"
APP_ICON_ERROR = "❌"
APP_ICON_SUCCESS = "✅"
APP_ICON_PROCESSING = "🧠"
APP_ICON_OPTIMIZING = "🎯"

# --- Audio Configuration ---
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHANNELS = 1
TEMP_FILE_PATH = "temp_recording.wav"

# --- Model Configuration ---
MODEL_MAP = {
    "Tiny (.en)": "openai/whisper-tiny.en",  # Real tiny model for ultra-fast processing
    "Small (.en)": "distil-whisper/distil-small.en", 
    "Medium (.en)": "distil-whisper/distil-medium.en",
    "Large (v3.5)": "distil-whisper/distil-large-v3.5",
}
# Use Medium for best balanced performance with Optimized mode
DEFAULT_MODEL_NAME = "Medium (.en)"  # Best balance with smart processing

# --- Smart Model Selection ---
# Automatically use different models based on recording length
ADAPTIVE_MODEL_SELECTION = False  # Disabled for consistency
MODEL_SELECTION_RULES = {
    # Recording length (seconds): Model to use
    0.0: "Tiny (.en)",     # Very short clips: use fastest
    3.0: "Small (.en)",    # Medium clips: balanced
    10.0: "Small (.en)",   # Longer clips: stick with Small for consistency
}

# --- Audio Quality Optimizations ---
ENABLE_AUDIO_NORMALIZATION = True
ENABLE_SILENCE_TRIMMING = True
SILENCE_THRESHOLD = 0.01  # Threshold for silence detection

# --- Generation Optimizations ---
GENERATION_PROFILES = {
    "speed": {
        "max_new_tokens": 64,
        "chunk_length_s": 10,
        "repetition_penalty": 1.0,
    },
    "balanced": {
        "max_new_tokens": 96,
        "chunk_length_s": 12, 
        "repetition_penalty": 1.1,
    },
    "accuracy": {
        "max_new_tokens": 128,
        "chunk_length_s": 15,
        "repetition_penalty": 1.2,
    }
}
DEFAULT_PROFILE = "balanced"

# --- Keyboard Configuration ---
# Available options: keyboard.Key.shift_r, keyboard.Key.shift_l, 
# keyboard.Key.space, keyboard.Key.ctrl, etc.
TRIGGER_KEY = keyboard.Key.shift_r

# --- Processing Configuration ---
MAX_RECORDING_DURATION = 15  # seconds (shorter for faster processing)
PASTE_DELAY = 0.3  # seconds to wait before pasting (faster)
CLIPBOARD_FALLBACK = True  # Copy to clipboard if paste fails
OPTIMIZE_FOR_SPEED = True  # Enable speed optimizations

# --- Voice Activity Detection (VAD) ---
ENABLE_VAD = True  # Enable smart speech detection
VAD_MODEL = "silero"  # Options: "silero", "energy", "webrtc"
VAD_THRESHOLD = 0.5  # Speech probability threshold (0.1-0.9)
VAD_MIN_SPEECH_DURATION = 0.3  # Minimum speech segment length (seconds)
VAD_MIN_SILENCE_DURATION = 0.5  # Minimum silence to split segments (seconds)
VAD_SPEECH_PAD_BEFORE = 0.3  # Padding before speech (seconds) - increased for better capture
VAD_SPEECH_PAD_AFTER = 0.2   # Padding after speech (seconds)
VAD_SHOW_FEEDBACK = True     # Show VAD detection in status bar

# --- Processing Mode Configuration ---
# Three clean processing modes:
# 1. Traditional: Proven VAD processing (baseline)
# 2. Optimized: Smart segmentation + user's model choice  
# 3. Ultra-Fast: Forces Tiny model + smart optimization

PROCESSING_MODE = "Optimized"  # Default to smart processing for best balance

# Smart Optimization Settings (used by Optimized and Ultra-Fast modes)
SMART_MAX_SEGMENT_LENGTH = 8.0  # Split long recordings at 8s for parallel processing
SMART_MIN_SEGMENT_LENGTH = 1.5  # Minimum segment size before merging
SMART_SEGMENT_OVERLAP_PADDING = 0.1  # Padding between segments to avoid cut words

# --- Performance Configuration ---
# Set to True to use smaller models for faster loading
FAST_MODE = False
if FAST_MODE:
    DEFAULT_MODEL_NAME = "Tiny (.en)"

# --- Cache Configuration ---
CACHE_DIR = os.path.expanduser("~/.whisperbar_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Speed Optimization Settings ---
# INT8 Quantization: 50-70% speed boost with 1-3% accuracy loss
ENABLE_QUANTIZATION = True  # Automatically applied to Ultra-Fast and Optimized modes

# Dynamic Sample Rate: 50% speed boost in Ultra-Fast mode  
ENABLE_DYNAMIC_SAMPLE_RATE = True  # Uses 8kHz for Ultra-Fast, 16kHz for others

# JIT Compilation: 20-30% speed boost
ENABLE_TORCH_COMPILE = True  # Use torch.compile() for faster inference

# VAD Optimization: Skip speech detection in Ultra-Fast mode
ENABLE_VAD_SKIP_ULTRA_FAST = True  # Process entire audio without VAD in Ultra-Fast

# Memory Pre-allocation: 5-15% speed boost
ENABLE_MEMORY_PREALLOCATION = True

# Streaming Processing: Real-time transcription (experimental)
ENABLE_STREAMING = False  # Enable for live transcription during recording

# --- Advanced Settings ---
# Torch dtype - use float16 for faster inference, float32 for compatibility
FORCE_FLOAT32 = False  # Set to True if you have issues with float16

# Audio preprocessing
ENABLE_NOISE_REDUCTION = False  # Requires noisereduce package
AUDIO_GAIN = 1.0  # Multiply audio volume by this factor

# UI Settings
SHOW_NOTIFICATIONS = True
NOTIFICATION_DURATION = 3  # seconds
HIDE_FROM_DOCK = True  # Set to False if you want the app to appear in dock 