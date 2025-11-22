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

# Audio Input Device Selection
# Always uses built-in MacBook microphone (input only, no system audio)
# AUTO_DETECT_BUILTIN_MIC will automatically find and use MacBook built-in microphone
AUTO_DETECT_BUILTIN_MIC = True  # Automatically use built-in MacBook microphone (RECOMMENDED)
AUDIO_INPUT_DEVICE = None  # None = auto-detect built-in mic, or device index (int) for manual selection
# Note: Only microphone input devices are used - system audio/output is never used

# Audio Handling Options
AUTO_PAUSE_MEDIA = True  # Automatically pause music/video when recording starts
RESUME_MEDIA_AFTER_RECORDING = True  # Resume media after recording stops
SHARE_MIC_DURING_CALLS = True  # Allow recording while in calls (may not work with all apps)
DETECT_AUDIO_CONFLICTS = True  # Warn if audio conflicts detected

# --- Model Configuration ---
MODEL_MAP = {
    "Tiny (.en)": "openai/whisper-tiny.en",  # Real tiny model for ultra-fast processing
    # Large, Medium, and Small models removed for faster, lighter operation
}
# Use Tiny for lightest model with Optimized mode
DEFAULT_MODEL_NAME = "Tiny (.en)"  # Lightest model for fastest processing

# --- Smart Model Selection ---
# Automatically use different models based on recording length
ADAPTIVE_MODEL_SELECTION = False  # Disabled for consistency
MODEL_SELECTION_RULES = {
    # Recording length (seconds): Model to use
    0.0: "Tiny (.en)",     # Very short clips: use fastest
    3.0: "Tiny (.en)",     # Medium clips: use fastest
    10.0: "Tiny (.en)",    # Longer clips: use fastest
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
# Customizable keyboard shortcuts (string format)
# Format: "alt_r" for single key, "cmd+shift+w" for combinations
# Available modifiers: cmd, shift, ctrl, alt, option
# Available keys: f1-f12, space, enter, a-z, 0-9, alt_l, alt_r, shift_l, shift_r, fn, etc.
TRIGGER_KEY = "alt_r"  # Primary push-to-talk key (hold to record)
TRIGGER_KEY_2 = "shift_l"  # Secondary push-to-talk key (left shift - hold to record)
# Both keys work simultaneously - you can use either alt_r or shift_l to trigger recording
# To disable secondary key, set to None or "none"
TOGGLE_RECORDING_KEY = "cmd+shift+w"  # Toggle recording on/off (without holding)
QUIT_APP_KEY = "cmd+shift+q"  # Quit application
RELOAD_MODEL_KEY = "cmd+shift+r"  # Reload current model
TOGGLE_APP_KEY = "cmd+shift+t"  # Enable/disable app (without quitting)

# --- Processing Configuration ---
MAX_RECORDING_DURATION = 15  # seconds (shorter for faster processing)
PASTE_DELAY = 0.15  # seconds to wait before pasting (ensures window is ready and cursor is positioned)
ENABLE_DIRECT_PASTE = True  # Direct paste at cursor position (PRODUCTION: Always enabled - paste directly at text cursor in any input field)
PASTE_RETRY_ATTEMPTS = 5  # Number of retry attempts for paste (increased for better reliability)
FORCE_DIRECT_PASTE = True  # Force direct paste - always try to paste at cursor position, not just copy to clipboard
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
VAD_SHOW_FEEDBACK = False    # Show VAD detection in status bar (PRODUCTION: Disabled for cleaner UI)

# --- Processing Mode Configuration ---
# Three clean processing modes:
# 1. Traditional: Proven VAD processing (baseline)
# 2. Optimized: Smart segmentation + user's model choice  
# 3. Ultra-Fast: Forces Tiny model + smart optimization

PROCESSING_MODE = "Optimized"

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
NOTIFICATION_DURATION = 2  # seconds (PRODUCTION: Reduced from 3 to 2)
HIDE_FROM_DOCK = True  # Set to False if you want the app to appear in dock

# --- Production Optimizations ---
# Reduce VAD feedback for cleaner production experience
VAD_SHOW_FEEDBACK = False  # Disable VAD feedback in status bar (PRODUCTION: Cleaner UI)

# --- Cloud API Configuration ---
USE_CLOUD_API = False  # Set to True to use cloud APIs instead of local models
CLOUD_PROVIDER = "openai"  # Options: "openai", "google", "deepgram", "custom"

# API Keys (set via environment variables or here - NOT RECOMMENDED for security)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY', '')

# Custom API Configuration
CUSTOM_API_URL = os.getenv('CUSTOM_API_URL', '')
CUSTOM_API_HEADERS = {}  # Additional headers as dict, e.g., {"X-Custom-Header": "value"}
CUSTOM_API_AUTH = "bearer"  # Options: "bearer", "api_key", "basic"

# Cloud API Timeout (seconds)
CLOUD_API_TIMEOUT = 30

# --- Memory and Battery Optimizations ---
ENABLE_LOW_POWER_MODE = True  # Reduces CPU usage when idle (PRODUCTION: Enabled for battery life)
LAZY_MODEL_LOADING = True  # Only load model when first recording starts (PRODUCTION: Faster startup)
AUTO_UNLOAD_MODELS = True  # Unload models when not in use to save memory (PRODUCTION: Enabled)
IDLE_QUEUE_CHECK_INTERVAL = 2.0  # Seconds between queue checks when idle (battery optimization)
DISABLE_BACKGROUND_PRELOADING = True  # Don't preload models in background (saves battery)

# --- Production Mode Settings ---
PRODUCTION_MODE = True  # Enable production optimizations (reduces logging, optimizes performance)
VERBOSE_LOGGING = False  # Set to False for production (reduces console output)
SHOW_DEBUG_INFO = False  # Hide debug information in production
ENABLE_PERFORMANCE_MONITORING = False  # Disable performance monitoring in production (saves resources) 