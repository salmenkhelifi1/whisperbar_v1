import rumps
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, BitsAndBytesConfig
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
import threading
import os
import sys
from pynput import keyboard
import pyperclip
import queue
# import noisereduce as nr # Keep this commented/removed based on previous decision

# Faster Whisper import (optional - falls back gracefully if not available)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    log_info("[Info] faster-whisper not available, using regular Whisper")

# Fix PATH for GUI applications (helps with ffmpeg detection)
os.environ['PATH'] = '/opt/homebrew/bin:/usr/local/bin:' + os.environ.get('PATH', '')

# --- Production Logging System ---
# Load production mode settings
try:
    from config import PRODUCTION_MODE, VERBOSE_LOGGING
    PRODUCTION_MODE = PRODUCTION_MODE if 'PRODUCTION_MODE' in dir() else True
    VERBOSE_LOGGING = VERBOSE_LOGGING if 'VERBOSE_LOGGING' in dir() else False
except:
    PRODUCTION_MODE = True  # Default to production mode
    VERBOSE_LOGGING = False

def log_info(msg, force=False):
    """Log info message only if verbose logging is enabled or forced."""
    if VERBOSE_LOGGING or force or not PRODUCTION_MODE:
        print(msg)

def log_debug(msg):
    """Log debug message only in non-production mode."""
    if not PRODUCTION_MODE:
        print(f"[Debug] {msg}")

def log_warning(msg):
    """Log warning message (always shown)."""
    print(f"[Warning] {msg}")

def log_error(msg):
    """Log error message (always shown)."""
    print(f"[Error] {msg}", file=sys.stderr)

# Global VAD model
vad_model = None

# Import Controller and time
from pynput.keyboard import Controller

# --- Constants and Configuration ---
APP_NAME = "WhisperBar"
# Using text-based icons for better macOS menu bar compatibility
APP_ICON_IDLE = "🎤 Mic"
APP_ICON_RECORDING = "🔴 Rec"
APP_ICON_TRANSCRIBING = "⚡ Trans"
APP_ICON_LOADING = "🚀 Load"
APP_ICON_ERROR = "❌ Error"
APP_ICON_SUCCESS = "✅ Ready"
APP_ICON_PROCESSING = "🧠 Proc"
APP_ICON_OPTIMIZING = "🎯 Opt"

SAMPLE_RATE = 16000  # Whisper expects 16kHz audio (dynamic based on mode)
CHANNELS = 1
TEMP_FILE_PATH = "temp_recording.wav"

# Smart Processing Configuration  
SMART_MAX_SEGMENT_LENGTH = 8.0  # Maximum segment length for optimization
SMART_MIN_SEGMENT_LENGTH = 1.5  # Minimum segment length before merging

# Model IDs mapping - Best models for accents (multilingual better than English-only)
MODEL_MAP = {
    # English-only models - Fast and lightweight (Large, Medium, and Small removed)
    "Tiny (.en)": "openai/whisper-tiny.en",  # Ultra-fast for Ultra-Fast mode
}

# Model metadata for UI display
MODEL_METADATA = {
    "Tiny (.en)": {"size": "39MB", "speed": "⚡⚡⚡", "accuracy": "🎯", "type": "local"},
}

# Auto-detect best model on startup (will be set in __init__)
DEFAULT_MODEL_NAME = None  # Will be set by detect_optimal_model()

# Use Faster Whisper by default (2-4x faster than regular Whisper)
USE_FASTER_WHISPER = True  # Set to False to use regular transformers Whisper

# Trigger keys - Will be loaded from config
TRIGGER_KEY = None  # Primary trigger key - Will be set by parse_key_shortcut()
TRIGGER_KEY_2 = None  # Secondary trigger key (optional) - Will be set by parse_key_shortcut()

# --- Global State ---
# Using queues for thread-safe communication
status_queue = queue.Queue()
transcription_queue = queue.Queue()

# Model caching for instant switching
cached_models = {}  # {model_id: {"pipe": pipeline, "processor": processor, "model": model}}
cached_faster_whisper_models = {}  # {model_id: WhisperModel instance}
preload_thread = None

# Transcription Pipeline related
transcription_pipe = None
faster_whisper_model = None  # Faster Whisper model instance
current_model_id = None
selected_model_name = DEFAULT_MODEL_NAME

# Recording related
is_recording = False
audio_frames = []
audio_stream = None
media_was_paused = False  # Track if media was paused for auto-resume

# Pre-allocated buffers for speed optimization (reduced for memory optimization)
try:
    from config import ENABLE_LOW_POWER_MODE
    if ENABLE_LOW_POWER_MODE:
        PRE_ALLOCATED_BUFFER_SIZE = 16000 * 15  # 15 seconds at 16kHz (reduced for battery)
    else:
        PRE_ALLOCATED_BUFFER_SIZE = 16000 * 30  # 30 seconds at 16kHz
except:
    PRE_ALLOCATED_BUFFER_SIZE = 16000 * 30  # Default

pre_allocated_audio_buffer = None  # Lazy allocation
temp_processing_buffer = None  # Lazy allocation

# Processing State
current_processing_mode = "Optimized"

# Keyboard listener related
listener_thread = None
keyboard_listener = None
trigger_key_held = False
app_enabled = True  # Can disable app without quitting

# App instance reference for menu updates
app_instance_global = None

# Processing Control State
is_processing = False
cancel_processing_event = threading.Event()

# Keyboard shortcut manager
shortcut_manager = None
modifier_keys_pressed = set()  # Track pressed modifier keys

# Streaming processing state
streaming_buffer = []
streaming_thread = None
streaming_enabled = False
STREAMING_CHUNK_DURATION = 2.0  # Process every 2 seconds

# Keyboard Controller
keyboard_controller = Controller() # Instantiate the controller

# --- Helper Functions ---
def load_vad_model():
    """Load the VAD model for speech detection."""
    global vad_model
    try:
        from config import ENABLE_VAD, VAD_MODEL
        if not ENABLE_VAD:
            log_info("[Info] VAD disabled in config")
            return None
            
        if VAD_MODEL == "silero":
            log_info("[Info] Loading Silero VAD model...")
            import torch
            # Download Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                verbose=False
            )
            vad_model = {'model': model, 'utils': utils}
            log_info("[Info] Silero VAD model loaded successfully")
            return vad_model
        else:
            log_warning(f"VAD model '{VAD_MODEL}' not implemented yet")
            return None
            
    except Exception as e:
        log_warning(f"Failed to load VAD model: {e}")
        return None

def detect_speech_segments(audio_data, sample_rate=16000):
    """Detect speech segments in audio using VAD."""
    try:
        from config import (VAD_THRESHOLD, VAD_MIN_SPEECH_DURATION, 
                           VAD_MIN_SILENCE_DURATION, VAD_SPEECH_PAD_BEFORE, 
                           VAD_SPEECH_PAD_AFTER)
        
        if vad_model is None:
            log_info("[Info] VAD not available, processing entire audio")
            return [(0, len(audio_data))]
            
        # Ensure audio is 1D and float32
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()
        audio_data = audio_data.astype(np.float32)
        
        # Convert to torch tensor
        import torch
        audio_tensor = torch.from_numpy(audio_data)
        
        # Get speech timestamps using Silero VAD
        get_speech_timestamps = vad_model['utils'][0]
        speech_timestamps = get_speech_timestamps(
            audio_tensor, 
            vad_model['model'],
            sampling_rate=sample_rate,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=int(VAD_MIN_SPEECH_DURATION * 1000),
            min_silence_duration_ms=int(VAD_MIN_SILENCE_DURATION * 1000),
            return_seconds=False
        )
        
        if not speech_timestamps:
            log_info("[Info] No speech detected by VAD")
            return []
            
        # Convert to sample indices and add padding
        segments = []
        for segment in speech_timestamps:
            start_sample = max(0, segment['start'] - int(VAD_SPEECH_PAD_BEFORE * sample_rate))
            end_sample = min(len(audio_data), segment['end'] + int(VAD_SPEECH_PAD_AFTER * sample_rate))
            segments.append((start_sample, end_sample))
            
        # Merge overlapping segments
        merged_segments = []
        for start, end in sorted(segments):
            if merged_segments and start <= merged_segments[-1][1]:
                # Merge with previous segment
                merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], end))
            else:
                merged_segments.append((start, end))
                
        total_speech_duration = sum((end - start) / sample_rate for start, end in merged_segments)
        total_duration = len(audio_data) / sample_rate
        speech_ratio = total_speech_duration / total_duration if total_duration > 0 else 0
        
        log_info(f"[Info] VAD detected {len(merged_segments)} speech segments")
        log_info(f"[Info] Speech ratio: {speech_ratio:.1%} ({total_speech_duration:.1f}s/{total_duration:.1f}s)")
        
        return merged_segments
        
    except Exception as e:
        log_warning(f"VAD processing failed: {e}")
        return [(0, len(audio_data))]  # Fallback to entire audio

# --- Smart Processing Functions ---
def process_audio_traditional(audio_data_np):
    """Traditional VAD processing (proven baseline method)."""
    try:
        log_info(f"[Traditional] Processing {len(audio_data_np)/SAMPLE_RATE:.1f}s of audio")
        
        # Use standard VAD to detect speech segments
        speech_segments = detect_speech_segments(audio_data_np)
        
        if not speech_segments:
            log_info("[Traditional] No speech detected")
            return ""
        
        # Process each speech segment and combine results
        transcriptions = []
        total_processed_duration = 0
        
        for i, (start_idx, end_idx) in enumerate(speech_segments):
            segment_audio = audio_data_np[start_idx:end_idx]
            segment_duration = len(segment_audio) / SAMPLE_RATE
            total_processed_duration += segment_duration
            
            log_info(f"[Traditional] Processing speech segment {i+1}/{len(speech_segments)} ({segment_duration:.1f}s)")
            
            # Transcribe this segment
            segment_result = transcription_pipe({
                "raw": segment_audio, 
                "sampling_rate": SAMPLE_RATE
            })
            
            if segment_result and "text" in segment_result:
                segment_text = segment_result["text"].strip()
                if segment_text:
                    transcriptions.append(segment_text)
        
        # Combine all transcriptions
        final_text = " ".join(transcriptions).strip() if transcriptions else ""
        final_text = " ".join(final_text.split())  # Clean up extra spaces
        
        # Calculate speed improvement
        original_duration = len(audio_data_np) / SAMPLE_RATE
        speed_improvement = ((original_duration - total_processed_duration) / original_duration * 100) if original_duration > 0 else 0
        log_info(f"[Traditional] VAD processing complete. Speed improvement: {speed_improvement:.1f}%")
        
        return final_text
        
    except Exception as e:
        log_error(f"Traditional processing failed: {e}")
        return ""

def process_audio_optimized(audio_data_np):
    """Optimized processing pipeline with smart segmentation."""
    try:
        from config import SMART_MAX_SEGMENT_LENGTH, SMART_MIN_SEGMENT_LENGTH
        
        start_time = time.time()
        log_info(f"[Optimized] Starting smart transcription for {len(audio_data_np)/SAMPLE_RATE:.1f}s of audio")
        
        # Smart segmentation with VAD
        speech_segments = detect_speech_segments(audio_data_np)
        if not speech_segments:
            log_info("[Optimized] No speech detected")
            return ""
        
        # Optimize segments for better processing
        optimized_segments = optimize_segments_for_speed(speech_segments, audio_data_np)
        
        # Process segments sequentially (thread-safe)
        final_text = process_segments_sequential(optimized_segments, audio_data_np)
        
        processing_time = time.time() - start_time
        log_info(f"[Optimized] Complete in {processing_time:.2f}s: '{final_text[:50]}...'")
        
        return final_text
        
    except Exception as e:
        log_error(f"Optimized processing failed: {e}")
        return ""

def process_audio_ultra_fast(audio_data_np):
    """Ultra-Fast processing pipeline optimized for maximum speed with Tiny model."""
    try:
        start_time = time.time()
        sample_rate = get_optimal_sample_rate()
        duration = len(audio_data_np)/sample_rate
        log_info(f"[Ultra-Fast] Starting maximum-speed transcription for {duration:.1f}s of audio")
        
        # Skip VAD entirely for maximum speed - just process everything as one segment
        log_info("[Ultra-Fast] Skipping VAD for maximum speed - processing entire audio")
        speech_segments = [(0, len(audio_data_np))]  # Process entire audio as one segment
        
        # Process segments with speed priority (bypassing optimization)
        final_text = process_segments_sequential(speech_segments, audio_data_np)
        
        processing_time = time.time() - start_time
        log_info(f"[Ultra-Fast] Complete in {processing_time:.2f}s: '{final_text[:50]}...' (Tiny model + no VAD)")
        
        return final_text
        
    except Exception as e:
        log_error(f"Ultra-fast processing failed: {e}")
        return ""

def optimize_segments_for_speed(speech_segments, audio_data_np):
    """Optimize VAD segments for better processing efficiency."""
    try:
        from config import SMART_MAX_SEGMENT_LENGTH, SMART_MIN_SEGMENT_LENGTH
        
        optimized = []
        
        for start_idx, end_idx in speech_segments:
            segment_duration = (end_idx - start_idx) / SAMPLE_RATE
            
            if segment_duration <= SMART_MAX_SEGMENT_LENGTH:
                # Segment is optimal size
                optimized.append((start_idx, end_idx))
            else:
                # Split long segments at natural pauses (using VAD)
                segment_audio = audio_data_np[start_idx:end_idx]
                sub_segments = detect_speech_segments(segment_audio)
                
                # Convert back to global indices and group into chunks
                current_chunk_start = start_idx
                for sub_start, sub_end in sub_segments:
                    global_start = start_idx + sub_start
                    global_end = start_idx + sub_end
                    
                    chunk_duration = (global_end - current_chunk_start) / SAMPLE_RATE
                    if chunk_duration >= SMART_MAX_SEGMENT_LENGTH:
                        # Finalize current chunk
                        if current_chunk_start < global_start:
                            optimized.append((current_chunk_start, global_start))
                        current_chunk_start = global_start
                
                # Add final chunk
                if current_chunk_start < end_idx:
                    optimized.append((current_chunk_start, end_idx))
        
        # Merge very small segments
        merged = []
        current_start = None
        current_end = None
        
        for start_idx, end_idx in optimized:
            duration = (end_idx - start_idx) / SAMPLE_RATE
            
            if duration < SMART_MIN_SEGMENT_LENGTH:
                if current_start is None:
                    current_start = start_idx
                current_end = end_idx
            else:
                # Finalize any pending small segment
                if current_start is not None:
                    merged.append((current_start, current_end))
                    current_start = None
                
                # Add this segment
                merged.append((start_idx, end_idx))
        
        # Add final small segment if exists
        if current_start is not None:
            merged.append((current_start, current_end))
        
        log_info(f"[Smart] Optimized {len(speech_segments)} segments → {len(merged)} segments")
        return merged
        
    except Exception as e:
        print(f"[Warning] Segment optimization failed: {e}")
        return speech_segments

def transcribe_segment_fast(segment_audio, segment_id):
    """Fast transcription of a single segment using optimized pipeline."""
    try:
        # Check if model is available (either Faster Whisper or regular)
        model_available = False
        if USE_FASTER_WHISPER and FASTER_WHISPER_AVAILABLE:
            model_available = (faster_whisper_model is not None)
        else:
            model_available = (transcription_pipe is not None)
        
        if not model_available:
            log_info(f"[Smart] Model not loaded for segment {segment_id}")
            return ""
        
        # Ensure correct format
        if segment_audio.ndim > 1:
            segment_audio = segment_audio.squeeze()
        segment_audio = segment_audio.astype(np.float32)
        
        duration = len(segment_audio) / SAMPLE_RATE
        log_info(f"[Smart] Transcribing segment {segment_id} ({duration:.1f}s)")
        
        start_time = time.time()
        result = transcription_pipe({
            "raw": segment_audio,
            "sampling_rate": SAMPLE_RATE
        })
        processing_time = time.time() - start_time
        
        text = result.get('text', '').strip() if result else ''
        log_info(f"[Smart] Segment {segment_id} done in {processing_time:.2f}s: '{text[:30]}...'")
        
        return text
        
    except Exception as e:
        print(f"[Error] Smart transcription failed for segment {segment_id}: {e}")
        return ""

def process_segments_sequential(segments, audio_data_np):
    """Process segments sequentially with smart optimizations."""
    try:
        log_info(f"[Smart] Processing {len(segments)} segments sequentially")
        
        results = []
        for i, (start_idx, end_idx) in enumerate(segments):
            segment_audio = audio_data_np[start_idx:end_idx]
            result = transcribe_segment_fast(segment_audio, i)
            if result:
                results.append(result)
        
        final_text = ' '.join(results).strip()
        final_text = ' '.join(final_text.split())  # Clean up whitespace
        
        return final_text
        
    except Exception as e:
        print(f"[Error] Sequential processing failed: {e}")
        return ""

def parse_key_shortcut(shortcut_str):
    """Parse keyboard shortcut string into pynput key object.
    
    Args:
        shortcut_str: String like "alt_r", "cmd+shift+w", "f1", etc.
        
    Returns:
        pynput keyboard.Key or str for regular keys
    """
    if not shortcut_str:
        return None
    
    shortcut_str = shortcut_str.lower().strip()
    
    # Map string names to pynput keys
    key_map = {
        'alt_r': keyboard.Key.alt_r,
        'alt_l': keyboard.Key.alt_l,
        'shift_r': keyboard.Key.shift_r,
        'shift_l': keyboard.Key.shift_l,
        'ctrl_r': keyboard.Key.ctrl_r,
        'ctrl_l': keyboard.Key.ctrl_l,
        'cmd': keyboard.Key.cmd,
        'space': keyboard.Key.space,
        'enter': keyboard.Key.enter,
        'tab': keyboard.Key.tab,
        'esc': keyboard.Key.esc,
        'backspace': keyboard.Key.backspace,
        'delete': keyboard.Key.delete,
        'up': keyboard.Key.up,
        'down': keyboard.Key.down,
        'left': keyboard.Key.left,
        'right': keyboard.Key.right,
        'home': keyboard.Key.home,
        'end': keyboard.Key.end,
        'page_up': keyboard.Key.page_up,
        'page_down': keyboard.Key.page_down,
        # Fn key support - try different approaches
        'fn': getattr(keyboard.Key, 'fn', None) or getattr(keyboard.Key, 'f24', None),  # Fn key (may not work on all keyboards)
        'f24': getattr(keyboard.Key, 'f24', None),  # Some keyboards map Fn to F24
        'menu': getattr(keyboard.Key, 'menu', None),  # Context menu key (alternative)
    }
    
    # Function keys
    for i in range(1, 13):
        key_map[f'f{i}'] = getattr(keyboard.Key, f'f{i}', None)
    
    # Handle combinations (e.g., "cmd+shift+w")
    if '+' in shortcut_str:
        parts = [p.strip() for p in shortcut_str.split('+')]
        modifiers = []
        main_key = None
        
        for part in parts:
            if part in ['cmd', 'shift', 'ctrl', 'alt', 'option']:
                if part == 'option':
                    part = 'alt'
                modifiers.append(part)
            else:
                # Main key
                if part in key_map:
                    main_key = key_map[part]
                elif len(part) == 1:
                    main_key = part  # Single character
                else:
                    main_key = key_map.get(part, part)
        
        return {'modifiers': modifiers, 'key': main_key}
    
    # Single key
    if shortcut_str in key_map:
        return key_map[shortcut_str]
    elif len(shortcut_str) == 1:
        return shortcut_str  # Single character
    else:
        # Try to find in key_map
        return key_map.get(shortcut_str, None)


class KeyboardShortcutManager:
    """Manages customizable keyboard shortcuts."""
    
    def __init__(self):
        self.shortcuts = {}
        self.modifier_keys = set()
        self.listener = None
        self.load_shortcuts()
    
    def load_shortcuts(self):
        """Load shortcuts from config."""
        try:
            from config import (
                TRIGGER_KEY, TRIGGER_KEY_2, TOGGLE_RECORDING_KEY, QUIT_APP_KEY,
                RELOAD_MODEL_KEY, TOGGLE_APP_KEY
            )
            
            self.shortcuts = {
                'trigger': parse_key_shortcut(TRIGGER_KEY),
                'trigger_2': parse_key_shortcut(TRIGGER_KEY_2) if TRIGGER_KEY_2 and TRIGGER_KEY_2.lower() != 'none' else None,
                'toggle_recording': parse_key_shortcut(TOGGLE_RECORDING_KEY),
                'quit': parse_key_shortcut(QUIT_APP_KEY),
                'reload': parse_key_shortcut(RELOAD_MODEL_KEY),
                'toggle_app': parse_key_shortcut(TOGGLE_APP_KEY),
            }
            
            print(f"[Shortcuts] Loaded shortcuts from config")
        except Exception as e:
            print(f"[Shortcuts] Error loading shortcuts: {e}")
            # Fallback to defaults
            self.shortcuts = {
                'trigger': keyboard.Key.alt_r,
                'toggle_recording': {'modifiers': ['cmd', 'shift'], 'key': 'w'},
                'quit': {'modifiers': ['cmd', 'shift'], 'key': 'q'},
                'reload': {'modifiers': ['cmd', 'shift'], 'key': 'r'},
                'toggle_app': {'modifiers': ['cmd', 'shift'], 'key': 't'},
            }
    
    def _check_modifiers(self, modifiers_needed):
        """Check if required modifiers are pressed."""
        if not modifiers_needed:
            return True
        
        pressed_modifiers = set()
        # Check cmd (meta key)
        try:
            import AppKit
            cmd_pressed = AppKit.NSEvent.modifierFlags() & AppKit.NSEventModifierFlagCommand
            if cmd_pressed:
                pressed_modifiers.add('cmd')
        except:
            pass
        
        # For now, use a simpler approach - track modifiers in global state
        # This is a simplified version - full implementation would track all modifiers
        return True  # Simplified for now
    
    def _matches_shortcut(self, key, shortcut_def):
        """Check if pressed key matches shortcut definition."""
        if shortcut_def is None:
            return False
        
        # Handle simple key objects
        if isinstance(shortcut_def, keyboard.Key):
            return key == shortcut_def
        
        # Handle string keys
        if isinstance(shortcut_def, str):
            try:
                key_char = key.char if hasattr(key, 'char') and key.char else None
                return key_char == shortcut_def.lower()
            except:
                return False
        
        # Handle combination shortcuts
        if isinstance(shortcut_def, dict) and 'key' in shortcut_def:
            main_key = shortcut_def['key']
            modifiers = shortcut_def.get('modifiers', [])
            
            # Check main key
            if isinstance(main_key, keyboard.Key):
                key_match = key == main_key
            elif isinstance(main_key, str):
                try:
                    key_char = key.char if hasattr(key, 'char') and key.char else None
                    key_match = key_char == main_key.lower()
                except:
                    key_match = False
            else:
                key_match = False
            
            # Check modifiers (simplified - would need full modifier tracking)
            modifiers_match = self._check_modifiers(modifiers)
            
            return key_match and modifiers_match
        
        return False
    
    def handle_key_press(self, key):
        """Handle key press event."""
        global app_enabled, is_recording, app_instance_global
        
        # Check if app is enabled
        if not app_enabled and not self._matches_shortcut(key, self.shortcuts.get('toggle_app')):
            return
        
        # Check each shortcut
        if self._matches_shortcut(key, self.shortcuts.get('trigger')):
            # Trigger key - handled by existing on_press
            return
        
        if self._matches_shortcut(key, self.shortcuts.get('toggle_recording')):
            if app_instance_global:
                app_instance_global.toggle_recording(None)
            return
        
        if self._matches_shortcut(key, self.shortcuts.get('reload')):
            if app_instance_global:
                app_instance_global.reload_model(None)
            return
        
        if self._matches_shortcut(key, self.shortcuts.get('toggle_app')):
            app_enabled = not app_enabled
            status_queue.put(f"{APP_ICON_SUCCESS if app_enabled else APP_ICON_IDLE} {'Enabled' if app_enabled else 'Disabled'}")
            return
        
        if self._matches_shortcut(key, self.shortcuts.get('quit')):
            if app_instance_global:
                app_instance_global.quit_application(None)
            return


def detect_optimal_model():
    """Auto-detects the best model based on Mac capabilities.
    
    Returns:
        str: Model name from MODEL_MAP that best fits the system
    """
    import platform
    import psutil
    
    try:
        # Detect Apple Silicon vs Intel
        is_apple_silicon = platform.processor() == 'arm' or 'arm64' in platform.machine().lower()
        has_mps = torch.backends.mps.is_available()
        
        # Check available RAM
        try:
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
        except:
            available_ram_gb = 8.0  # Default assumption
        
        # Check GPU availability
        has_gpu = has_mps or torch.cuda.is_available()
        
        print(f"[Auto-Detect] System Info:")
        print(f"  - Apple Silicon: {is_apple_silicon}")
        print(f"  - MPS Available: {has_mps}")
        print(f"  - Available RAM: {available_ram_gb:.1f} GB")
        print(f"  - GPU Available: {has_gpu}")
        
        # Decision logic - Only Tiny model available now
        optimal_model = "Tiny (.en)"  # Only model available
        print(f"[Auto-Detect] ✅ Selected: {optimal_model} (Only model available)")
        
        return optimal_model
        
    except Exception as e:
        print(f"[Auto-Detect] Error detecting optimal model: {e}")
        # Safe fallback
        return "Tiny (.en)"

def get_device_and_dtype():
    """Determines the best device (MPS, CUDA, CPU) and appropriate dtype."""
    if torch.backends.mps.is_available():
        device = "mps"
        # Use float16 for better speed on Apple Silicon
        try:
            torch_dtype = torch.float16
            print(f"[Info] Using float16 for speed optimization on MPS")
        except:
            torch_dtype = torch.float32
            print(f"[Info] Fallback to float32 on MPS")
    elif torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32
    print(f"Selected device: {device} with dtype: {torch_dtype}")
    return device, torch_dtype

def get_quantization_config(processing_mode=None):
    """Get quantization configuration based on processing mode for speed optimization."""
    try:
        if processing_mode is None:
            from config import PROCESSING_MODE
            processing_mode = PROCESSING_MODE
    except:
        processing_mode = "Optimized"  # Default
    
    # Skip quantization on macOS/MPS - bitsandbytes requires CUDA
    if torch.backends.mps.is_available():
        print(f"[Quantization] Skipping quantization on macOS/MPS (bitsandbytes requires CUDA)")
        return None
    
    # Use INT8 quantization for speed-focused modes (50-70% speed boost, 1-3% accuracy loss)
    # Only on CUDA systems
    if processing_mode in ["Ultra-Fast", "Optimized"]:
        if torch.cuda.is_available():
            print(f"[Quantization] Using INT8 quantization for {processing_mode} mode (50-70% speed boost)")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,  # More accurate quantization
            )
        else:
            print(f"[Quantization] CUDA not available, skipping quantization")
            return None
    else:
        print(f"[Quantization] Using full precision for {processing_mode} mode")
        return None

def get_optimal_sample_rate():
    """Get optimal sample rate based on processing mode for speed optimization."""
    try:
        from config import PROCESSING_MODE
        processing_mode = PROCESSING_MODE
    except:
        processing_mode = "Optimized"
    
    if processing_mode == "Ultra-Fast":
        print("[Optimization] Using 8kHz sample rate for Ultra-Fast mode (50% speed boost)")
        return 8000  # Half the data to process
    else:
        return 16000  # Standard Whisper sample rate

def update_config_value(key, value):
    """Update a configuration value in config.py file."""
    try:
        config_path = "config.py"
        
        # Read current config
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        # Update the specific line
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key} ="):
                lines[i] = f"{key} = {value}\n"
                updated = True
                break
        
        if updated:
            # Write back to file
            with open(config_path, 'w') as f:
                f.writelines(lines)
            
            # Reload the config module
            import importlib
            import config
            importlib.reload(config)
            
            print(f"[Config] Updated {key} = {value}")
        else:
            print(f"[Warning] Could not find {key} in config.py")
            
    except Exception as e:
        print(f"[Error] Failed to update config: {e}")

def load_faster_whisper_model(model_id, cache_only=False):
    """Loads Faster Whisper model (2-4x faster than regular Whisper).
    
    Args:
        model_id: The model identifier (e.g., "base.en", "medium.en")
        cache_only: If True, only cache the model without setting as active
    """
    global faster_whisper_model, current_model_id
    
    if not FASTER_WHISPER_AVAILABLE:
        print("[Warning] Faster Whisper not available, falling back to regular Whisper")
        return False
    
    try:
        # Check if model is already cached
        if model_id in cached_faster_whisper_models and not cache_only:
            print(f"[Faster Whisper] ⚡ Using cached model: {model_id}")
            faster_whisper_model = cached_faster_whisper_models[model_id]
            current_model_id = model_id
            status_queue.put(f"{APP_ICON_SUCCESS} Ready • {selected_model_name} • ⚡ Faster Whisper")
            return True
        
        if not cache_only:
            status_queue.put(f"{APP_ICON_LOADING} Loading Faster Whisper {selected_model_name}...")
        
        # Convert model_id format to Faster Whisper model names
        # Faster Whisper uses simpler names: "large-v3", "medium", "base.en", "medium.en"
        if "/" in model_id:
            model_name = model_id.split("/")[-1]  # Get "whisper-large-v3" or "whisper-base.en"
            if model_name.startswith("whisper-"):
                model_name = model_name.replace("whisper-", "")  # "large-v3" or "base.en"
            elif model_name.startswith("distil-"):
                model_name = model_name.replace("distil-", "")  # "medium.en"
        else:
            model_name = model_id
        
        # Faster Whisper model names are: tiny, base, small, medium, large-v2, large-v3
        # Keep .en suffix for English-only models, remove for multilingual
        # Examples: "large-v3" (multilingual), "base.en" (English-only)
        
        print(f"[Faster Whisper] Loading model: {model_name} (from {model_id})")
        
        # Determine device for Faster Whisper
        device = "cpu"
        compute_type = "int8"  # Faster with good accuracy
        
        if torch.backends.mps.is_available():
            device = "cpu"  # Faster Whisper doesn't support MPS directly, but CPU is fast
            compute_type = "int8"
            print("[Faster Whisper] Using CPU with int8 (MPS not directly supported)")
        elif torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            print("[Faster Whisper] Using CUDA with float16")
        else:
            device = "cpu"
            compute_type = "int8"
            print("[Faster Whisper] Using CPU with int8")
        
        # Load Faster Whisper model
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        
        # Cache the model
        cached_faster_whisper_models[model_id] = model
        
        # Update global state if not cache_only
        if not cache_only:
            faster_whisper_model = model
            current_model_id = model_id
            status_queue.put(f"{APP_ICON_SUCCESS} Ready • {selected_model_name} • ⚡ Faster Whisper")
        
        print(f"[Faster Whisper] ✅ Model {model_name} loaded successfully")
        return True
        
    except Exception as e:
        print(f"[Error] Failed to load Faster Whisper model: {e}")
        import traceback
        traceback.print_exc()
        if not cache_only:
            status_queue.put(f"{APP_ICON_ERROR} Faster Whisper Load Failed")
        return False

def load_model_and_processor(model_id, cache_only=False):
    """Loads the specified model and processor with caching support.
    Supports both Faster Whisper (faster) and regular Whisper (fallback).
    
    Args:
        model_id: The model identifier
        cache_only: If True, only cache the model without setting as active
    """
    global transcription_pipe, faster_whisper_model, current_model_id
    
    # Check if we should use Faster Whisper
    if USE_FASTER_WHISPER and FASTER_WHISPER_AVAILABLE:
        print("[Info] Using Faster Whisper (2-4x faster)")
        if load_faster_whisper_model(model_id, cache_only):
            return True
        else:
            print("[Warning] Faster Whisper failed, falling back to regular Whisper")
    
    # Fallback to regular Whisper
    try:
        # Check if Ultra-Fast mode should override model selection
        if not cache_only:
            try:
                from config import PROCESSING_MODE
                if PROCESSING_MODE == "Ultra-Fast" and model_id != MODEL_MAP.get("Tiny (.en)"):
                    original_model = model_id
                    model_id = MODEL_MAP.get("Tiny (.en)", model_id)
                    print(f"[Ultra-Fast] Overriding model {original_model} → {model_id} for maximum speed")
            except:
                pass
            
        # Check if model is already cached
        cached_result = get_cached_model(model_id)
        if cached_result and not cache_only:
            print(f"[Model] ⚡ Using cached model: {model_id}")
            transcription_pipe = cached_result["pipe"]
            current_model_id = model_id
            # Show ready status for cached model
            try:
                from config import PROCESSING_MODE
                mode_emoji = {"Optimized": "🎯", "Ultra-Fast": "⚡", "Traditional": "📊"}.get(PROCESSING_MODE, "📊")
                status_queue.put(f"{APP_ICON_SUCCESS} Ready • {selected_model_name} • {mode_emoji} {PROCESSING_MODE}")
            except:
                status_queue.put(f"{APP_ICON_SUCCESS} Ready • {selected_model_name}")
            return True
        elif cached_result and cache_only:
            return cached_result
            
        if not cache_only:
            status_queue.put(f"{APP_ICON_LOADING} Loading {selected_model_name}...")
        else:
            print(f"[Preload] Loading {model_id} for caching...")
        
        device, torch_dtype = get_device_and_dtype()

        # Get quantization config based on processing mode for speed optimization
        quantization_config = get_quantization_config()

        # Model loading parameters
        # For MPS and large models, use device_map to avoid meta tensor issues
        is_large_model = "large" in model_id.lower()
        
        model_kwargs = {
            "dtype": torch_dtype,  # Changed from torch_dtype to dtype (deprecation fix)
            "use_safetensors": True,
            "attn_implementation": "sdpa"  # Explicitly enable SDPA
        }
        
        # For large models on MPS, use device_map to avoid meta tensor issues
        if device == "mps" and is_large_model:
            print("[Info] Using device_map for large model on MPS to avoid meta tensor issues")
            model_kwargs["device_map"] = device
            # Don't use low_cpu_mem_usage with device_map
        elif device != "mps":
            # For CUDA or CPU, use low_cpu_mem_usage
            model_kwargs["low_cpu_mem_usage"] = True
        # For MPS with smaller models, don't use low_cpu_mem_usage or device_map
        
        # Add quantization if enabled
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            # Don't move to device manually when using quantization - it's handled automatically
            print("[Quantization] Loading model with INT8 quantization for speed boost...")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **model_kwargs)
        
        # Only move to device if not using quantization and not using device_map
        # device_map handles device placement automatically
        if quantization_config is None and "device_map" not in model_kwargs:
            try:
                # Check if model is actually loaded (not meta tensor)
                first_param = next(model.parameters())
                current_device = first_param.device.type
                
                # Try to access actual data to verify it's not a meta tensor
                try:
                    _ = first_param.data
                    has_data = True
                except (RuntimeError, AttributeError):
                    has_data = False
                    print("[Warning] Model appears to be meta tensor")
                
                if has_data and current_device == "cpu" and device != "cpu":
                    # Model has data and is on CPU, safe to move
                    print(f"[Info] Moving model from CPU to {device}...")
                    model = model.to(device)
                    print(f"[Info] ✅ Model moved to {device}")
                elif current_device != "cpu":
                    print(f"[Info] Model already on device: {current_device}")
                elif not has_data:
                    # Meta tensor detected - this shouldn't happen if device_map was used
                    print("[Warning] Meta tensor detected - model may not work correctly")
            except Exception as e:
                print(f"[Warning] Device placement check failed: {e}")
                # Model will be used as-is, pipeline will handle device
        elif "device_map" in model_kwargs:
            print(f"[Info] Model loaded with device_map, device placement handled automatically")
        
        # Apply torch.compile() for 20-30% speed boost (JIT compilation)
        # Skip for large models on MPS as it can cause issues
        try:
            if hasattr(torch, 'compile'):
                # Don't compile large models on MPS - they can cause memory/performance issues
                model_size = sum(p.numel() for p in model.parameters())
                is_large_model = "large" in model_id.lower() or model_size > 500_000_000
                
                if device == "mps" and is_large_model:
                    print("[Optimization] Skipping torch.compile() for large model on MPS (can cause issues)")
                else:
                    print("[Optimization] Applying torch.compile() for JIT acceleration...")
                    model = torch.compile(model, mode="reduce-overhead")
                    print("[Optimization] ✅ torch.compile() applied successfully")
        except Exception as e:
            print(f"[Optimization] torch.compile() not available or failed: {e}")
        processor = AutoProcessor.from_pretrained(model_id)

        # Conditionally set generation kwargs based on model type
        gen_kwargs = {}
        if not model_id.endswith(".en"):
            # For multilingual models, don't force language to allow transcription in original language
            # gen_kwargs["language"] = "english"  # Removed: This forces translation to English
            print(f"Using multilingual model: {model_id} - will transcribe in original language")
        else:
            print(f"Using English-only model: {model_id} - English transcription only")

        # Optimized generation parameters based on processing mode
        try:
            from config import PROCESSING_MODE
            processing_mode = PROCESSING_MODE
        except:
            processing_mode = "Optimized"
        
        if processing_mode == "Ultra-Fast":
            # Maximum speed settings
            optimized_gen_kwargs = {
                **gen_kwargs,
                "max_new_tokens": 64,      # Shorter for speed
                "num_beams": 1,            # Greedy decoding 
                "do_sample": False,        # Deterministic
                "use_cache": True,         # Enable KV caching
                "temperature": 0.0,        # No randomness
                "repetition_penalty": 1.0, # No penalty for speed
                "length_penalty": 0.8,     # Prefer shorter outputs
            }
        elif processing_mode == "Optimized":
            # Balanced speed + accuracy
            optimized_gen_kwargs = {
            **gen_kwargs,
            "max_new_tokens": 96,      # Balance between speed and completeness
            "num_beams": 1,            # Greedy decoding for speed
            "do_sample": False,        # Deterministic output
            "use_cache": True,         # Enable KV caching
            "temperature": 0.0,        # Deterministic (no randomness)
                "repetition_penalty": 1.05, # Light penalty to reduce repetition
            "length_penalty": 1.0,     # No length bias
            }
        else:  # Traditional
            # Accuracy-focused settings
            optimized_gen_kwargs = {
                **gen_kwargs,
                "max_new_tokens": 128,     # Allow longer outputs
                "num_beams": 2,            # Light beam search for better quality
                "do_sample": False,        # Deterministic output
                "use_cache": True,         # Enable KV caching
                "temperature": 0.0,        # Deterministic
                "repetition_penalty": 1.1, # Standard penalty
                "length_penalty": 1.0,     # No length bias
        }
        
        # For multilingual models, ensure no forced language or task tokens
        if not model_id.endswith(".en"):
            # Remove any language-forcing parameters that might have been inherited
            optimized_gen_kwargs.pop("language", None)
            optimized_gen_kwargs.pop("task", None)
            optimized_gen_kwargs.pop("forced_decoder_ids", None)
            print(f"[Info] Multilingual model: removed language forcing parameters")

        # Create pipeline with explicit multilingual support
        # Don't pass device if model is already on device (accelerate handles it)
        pipeline_kwargs = {
            "model": model,
            "tokenizer": processor.tokenizer,
            "feature_extractor": processor.feature_extractor,
            "chunk_length_s": 12,         # Optimal chunk size for balance
            "batch_size": 1,
            "return_timestamps": False,
            "dtype": torch_dtype,  # Changed from torch_dtype to dtype (deprecation fix)
            "generate_kwargs": optimized_gen_kwargs
        }
        
        # For MPS, don't pass device to pipeline - the model is already on MPS
        # Pipeline will automatically detect the model's device
        try:
            model_device = next(model.parameters()).device.type
            if model_device == "cpu":
                # Model is on CPU, let pipeline handle device placement
                pipeline_kwargs["device"] = device
            # If model is on MPS/CUDA, don't pass device - pipeline detects it automatically
            print(f"[Info] Pipeline will use model's device: {model_device}")
        except Exception as e:
            print(f"[Warning] Could not detect model device: {e}")
            # Fallback: pass device to pipeline
            pipeline_kwargs["device"] = device
        
        # For multilingual models, explicitly disable language forcing
        if not model_id.endswith(".en"):
            # Remove the task conflict - automatic-speech-recognition is already the task
            print(f"[Info] Multilingual model: ensuring no language forcing for {model_id}")
        
        pipe = pipeline("automatic-speech-recognition", **pipeline_kwargs)
        
        # Prepare result for caching
        result = {
            "pipe": pipe,
            "processor": processor,
            "model": model
        }
        
        # Cache the model
        cached_models[model_id] = result
        
        # Update global state if not cache_only
        if not cache_only:
            transcription_pipe = pipe
            current_model_id = model_id
        
        # Show current mode in status with enhanced info
        try:
            from config import PROCESSING_MODE
            mode_info = PROCESSING_MODE
            if mode_info == "Optimized":
                mode_emoji = "🎯"
            elif mode_info == "Ultra-Fast":
                mode_emoji = "⚡"
            else:
                mode_emoji = "📊"
            status_queue.put(f"{APP_ICON_SUCCESS} Ready • {selected_model_name} • {mode_emoji} {mode_info}")
        except:
            status_queue.put(f"{APP_ICON_SUCCESS} Ready • {selected_model_name}")
            
        print(f"Model {model_id} loaded successfully on {device}. Cached: {len(cached_models)} models")
        return result if cache_only else True

    except Exception as e:
        print(f"Error loading model {model_id}: {e}", file=sys.stderr)
        if not cache_only:
            status_queue.put(f"{APP_ICON_ERROR} Model Load Failed")
            transcription_pipe = None # Ensure pipe is None on failure
            current_model_id = None
            return False

def transcribe_audio_thread(audio_data_np):
    """Transcribes audio in a separate thread."""
    global transcription_pipe, faster_whisper_model, is_processing

    is_processing = True

    # Check if cloud API should be used
    try:
        from config import USE_CLOUD_API, CLOUD_PROVIDER
        use_cloud = USE_CLOUD_API
        cloud_provider_name = CLOUD_PROVIDER
    except:
        use_cloud = False
        cloud_provider_name = None
    
    # Try cloud API first if enabled
    if use_cloud:
        try:
            from cloud_apis import get_cloud_provider, CloudTranscriptionError
            from config import (
                OPENAI_API_KEY, GOOGLE_API_KEY, DEEPGRAM_API_KEY,
                CUSTOM_API_URL, CUSTOM_API_HEADERS, CUSTOM_API_AUTH
            )
            
            print(f"[Cloud] Using {cloud_provider_name} API for transcription...")
            status_queue.put(f"{APP_ICON_TRANSCRIBING} ☁️ {cloud_provider_name}...")
            
            # Get provider instance
            provider_kwargs = {}
            if cloud_provider_name == "openai":
                provider_kwargs['api_key'] = OPENAI_API_KEY
            elif cloud_provider_name == "google":
                provider_kwargs['api_key'] = GOOGLE_API_KEY
            elif cloud_provider_name == "deepgram":
                provider_kwargs['api_key'] = DEEPGRAM_API_KEY
            elif cloud_provider_name == "custom":
                provider_kwargs['api_url'] = CUSTOM_API_URL
                provider_kwargs['api_key'] = os.getenv('CUSTOM_API_KEY', '')
                provider_kwargs['headers'] = CUSTOM_API_HEADERS
                provider_kwargs['auth_type'] = CUSTOM_API_AUTH
            
            provider = get_cloud_provider(cloud_provider_name, **provider_kwargs)
            
            # Transcribe using cloud API
            start_time = time.time()
            transcription = provider.transcribe(audio_data_np, SAMPLE_RATE)
            processing_time = time.time() - start_time
            
            if transcription:
                transcription_queue.put(transcription)
                status_queue.put(f"{APP_ICON_SUCCESS} ✅ {len(transcription)} chars • {processing_time:.1f}s • ☁️")
                is_processing = False
                return
            else:
                print("[Cloud] Empty transcription from cloud API, falling back to local...")
                # Fall through to local transcription
                
        except CloudTranscriptionError as e:
            print(f"[Cloud] Cloud API failed: {e}, falling back to local model...")
            status_queue.put(f"{APP_ICON_ERROR} Cloud Failed, Using Local")
            # Fall through to local transcription
        except Exception as e:
            print(f"[Cloud] Cloud API error: {e}, falling back to local model...")
            import traceback
            traceback.print_exc()
            status_queue.put(f"{APP_ICON_ERROR} Cloud Error, Using Local")
            # Fall through to local transcription

    # Check if Faster Whisper is available and should be used
    use_faster_whisper = USE_FASTER_WHISPER and FASTER_WHISPER_AVAILABLE and faster_whisper_model is not None
    
    if not use_faster_whisper and transcription_pipe is None:
        print("[Error] Transcription pipe not available.", file=sys.stderr)
        status_queue.put(f"{APP_ICON_ERROR} No Model Loaded")
        is_processing = False
        return

    try:
        # --- Ensure correct dtype before saving ---
        audio_data_float32 = audio_data_np.astype(np.float32)

        # Skip file saving for direct processing (faster)
        if cancel_processing_event.is_set():
            print("[Info] Processing cancelled before transcription.")
            return

        # --- Transcription Stage ---
        if cancel_processing_event.is_set():
            print("[Info] Processing cancelled before transcription.")
            status_queue.put(f"{APP_ICON_IDLE} Ready ({selected_model_name})")
            return
        print("[Info] Starting transcription...")
        status_queue.put(f"{APP_ICON_TRANSCRIBING} Transcribing...")
        start_time_tx = time.time()

        # Check for cancellation during transcription
        if cancel_processing_event.is_set():
            print("[Info] Processing cancelled during transcription setup.")
            status_queue.put(f"{APP_ICON_IDLE} Ready ({selected_model_name})")
            return

        # Use Faster Whisper if available
        if use_faster_whisper:
            try:
                # Ensure single channel and correct shape
                if audio_data_float32.ndim > 1:
                    audio_data_float32 = audio_data_float32.squeeze()
                
                # Normalize audio
                audio_max = np.abs(audio_data_float32).max()
                if audio_max > 0:
                    audio_data_float32 = audio_data_float32 / audio_max * 0.95
                
                print("[Faster Whisper] Transcribing with Faster Whisper (2-4x faster)...")
                status_queue.put(f"{APP_ICON_TRANSCRIBING} ⚡ Faster Whisper...")
                
                # Faster Whisper expects (samples,) numpy array at 16kHz
                segments, info = faster_whisper_model.transcribe(
                    audio_data_float32,
                    beam_size=1,  # Greedy decoding for speed
                    language="en",  # English-only
                    task="transcribe",
                    vad_filter=True,  # Use VAD for better quality
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Collect all segments
                transcription_parts = []
                for segment in segments:
                    transcription_parts.append(segment.text.strip())
                
                transcription = " ".join(transcription_parts).strip()
                
                processing_time = time.time() - start_time_tx
                audio_duration = len(audio_data_float32) / SAMPLE_RATE
                print(f"[Faster Whisper] ✅ Transcribed {audio_duration:.1f}s in {processing_time:.2f}s (speed: {audio_duration/processing_time:.1f}x)")
                
                if transcription:
                    transcription_queue.put(transcription)
                    status_queue.put(f"{APP_ICON_SUCCESS} ✅ {len(transcription)} chars • {processing_time:.1f}s")
                else:
                    status_queue.put(f"{APP_ICON_ERROR} No transcription")
                
                is_processing = False
                return
                
            except Exception as e:
                print(f"[Error] Faster Whisper transcription failed: {e}")
                import traceback
                traceback.print_exc()
                print("[Info] Falling back to regular Whisper...")
                # Continue to regular Whisper fallback below

        # Enhanced audio preprocessing with smart processing modes (regular Whisper)
        try:
            # Ensure single channel and correct shape
            if audio_data_float32.ndim > 1:
                audio_data_float32 = audio_data_float32.squeeze()
            
            # Audio quality optimizations
            if len(audio_data_float32) > 0:
                # Normalize audio to prevent clipping issues
                audio_max = np.abs(audio_data_float32).max()
                if audio_max > 0:
                    audio_data_float32 = audio_data_float32 / audio_max * 0.95
                
                # Determine processing mode
                try:
                    from config import PROCESSING_MODE
                    processing_mode = PROCESSING_MODE
                except:
                    processing_mode = "Optimized"  # Default to optimized
                
                # Process based on selected mode - with comprehensive fallback
                transcription = None
                audio_duration = len(audio_data_float32) / SAMPLE_RATE
                
                if processing_mode == "Ultra-Fast":
                    # Ultra-Fast Mode: Force Tiny model + optimized processing
                    try:
                        status_queue.put(f"⚡ Ultra-Fast • {audio_duration:.1f}s → Processing...")
                        transcription = process_audio_ultra_fast(audio_data_float32)
                    except Exception as ultra_fast_error:
                        print(f"[Warning] Ultra-fast mode failed: {ultra_fast_error}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to simple direct transcription
                        try:
                            print("[Info] Trying simple direct transcription...")
                            result = transcription_pipe({
                                "raw": audio_data_float32,
                                "sampling_rate": SAMPLE_RATE
                            })
                            transcription = result.get("text", "").strip() if result else None
                        except Exception as e:
                            print(f"[Error] Simple transcription failed: {e}")
                            transcription = None
                
                elif processing_mode == "Optimized":
                    # Optimized Mode: Smart processing + user's model choice
                    try:
                        status_queue.put(f"🎯 Optimized • {audio_duration:.1f}s → Smart Processing...")
                        transcription = process_audio_optimized(audio_data_float32)
                    except Exception as optimized_error:
                        print(f"[Warning] Optimized mode failed: {optimized_error}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to simple direct transcription
                        try:
                            print("[Info] Trying simple direct transcription...")
                            result = transcription_pipe({
                                "raw": audio_data_float32,
                                "sampling_rate": SAMPLE_RATE
                            })
                            transcription = result.get("text", "").strip() if result else None
                        except Exception as e:
                            print(f"[Error] Simple transcription failed: {e}")
                            transcription = None
                
                elif processing_mode == "Traditional":
                    # Traditional Mode: Proven baseline VAD processing  
                    try:
                        status_queue.put(f"📊 Traditional • {audio_duration:.1f}s → Processing...")
                        transcription = process_audio_traditional(audio_data_float32)
                    except Exception as traditional_error:
                        print(f"[Warning] Traditional mode failed: {traditional_error}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to simple direct transcription
                        try:
                            print("[Info] Trying simple direct transcription...")
                            result = transcription_pipe({
                                "raw": audio_data_float32,
                                "sampling_rate": SAMPLE_RATE
                            })
                            transcription = result.get("text", "").strip() if result else None
                        except Exception as e:
                            print(f"[Error] Simple transcription failed: {e}")
                            transcription = None
                
                # Final fallback if transcription is still None
                if not transcription:
                    print("[Info] All methods failed, trying final fallback...")
                    try:
                        result = transcription_pipe({
                            "raw": audio_data_float32,
                            "sampling_rate": SAMPLE_RATE
                        })
                        transcription = result.get("text", "").strip() if result else None
                        if transcription:
                            print(f"[Info] ✅ Fallback transcription succeeded: '{transcription[:50]}...'")
                    except Exception as final_error:
                        print(f"[Error] Final fallback failed: {final_error}")
                        import traceback
                        traceback.print_exc()
                        transcription = None
                
                print(f"[Info] Audio processing successful")
            else:
                raise ValueError("Empty audio data")
                
        except Exception as e:
            print(f"[Info] Direct numpy failed: {e}")
            print("[Info] Falling back to file-based transcription...")
            
            # Only save file when needed for fallback
            try:
                wav.write(TEMP_FILE_PATH, SAMPLE_RATE, audio_data_float32)
                result = transcription_pipe(TEMP_FILE_PATH)
                transcription = result["text"].strip() if result and "text" in result else None
            except Exception as e2:
                print(f"[Error] File-based transcription also failed: {e2}")
                transcription = None

        end_time_tx = time.time()

        if cancel_processing_event.is_set():
            print("[Info] Processing cancelled - transcription result discarded.")
            status_queue.put(f"{APP_ICON_IDLE} Ready ({selected_model_name})")
            return

        if transcription:
            processing_time = end_time_tx - start_time_tx
            print(f"[Info] Transcription finished ({processing_time:.2f}s). Result:")
            print(f"---> {transcription}")
            # Show success with timing info
            status_queue.put(f"{APP_ICON_SUCCESS} Done in {processing_time:.1f}s • {len(transcription)} chars")
            transcription_queue.put(transcription)
        else:
            print("[Warning] Transcription failed or produced empty result.", file=sys.stderr)
            status_queue.put(f"{APP_ICON_ERROR} No speech detected")

    except Exception as e:
        if not cancel_processing_event.is_set():
            print(f"[Error] Unexpected error during processing: {e}", file=sys.stderr)
            status_queue.put(f"{APP_ICON_ERROR} Failed")
        else:
            print(f"[Info] Processing thread terminated due to cancellation signal.")
    finally:
        # --- Cleanup Stage ---
        if os.path.exists(TEMP_FILE_PATH):
            try:
                os.remove(TEMP_FILE_PATH)
                if not cancel_processing_event.is_set():
                     print("[Info] Temporary audio file cleaned up.")
            except Exception as e:
                print(f"[Error] Deleting temp file failed: {e}", file=sys.stderr)

        is_processing = False

        if not cancel_processing_event.is_set() and status_queue.empty():
            status_queue.put(f"{APP_ICON_IDLE} Ready ({selected_model_name})")
        print("[Info] Processing thread finished.")


def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio block - with error handling."""
    try:
        if status:
            print(f"Audio Stream Status: {status}", file=sys.stderr)
        if is_recording:
            try:
                audio_frames.append(indata.copy())
            except Exception as e:
                print(f"[Error] Failed to append audio frame: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[Error] Audio callback error: {e}", file=sys.stderr)

def pause_media_playback():
    """Pause media playback (Spotify, YouTube, etc.) using AppleScript."""
    try:
        import subprocess
        # Try to pause Spotify
        spotify_script = '''
            tell application "Spotify"
                if player state is playing then
                    pause
                    return true
                end if
            end tell
            return false
        '''
        result = subprocess.run(['osascript', '-e', spotify_script], 
                              capture_output=True, timeout=1)
        if result.returncode == 0 and b'true' in result.stdout:
            log_info("[Audio] Paused Spotify")
            return True
        
        # Try to pause YouTube (Chrome/Safari)
        youtube_script = '''
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is true
                if frontApp is "Google Chrome" or frontApp is "Safari" then
                    keystroke "k" using {command down}
                    return true
                end if
            end tell
            return false
        '''
        result = subprocess.run(['osascript', '-e', youtube_script], 
                              capture_output=True, timeout=1)
        if result.returncode == 0:
            log_info("[Audio] Paused media playback")
            return True
        
        # Try to pause Music app
        music_script = '''
            tell application "Music"
                if player state is playing then
                    pause
                    return true
                end if
            end tell
            return false
        '''
        result = subprocess.run(['osascript', '-e', music_script], 
                              capture_output=True, timeout=1)
        if result.returncode == 0 and b'true' in result.stdout:
            log_info("[Audio] Paused Music app")
            return True
        
        return False
    except Exception as e:
        log_info(f"[Audio] Could not pause media: {e}")
        return False

def resume_media_playback():
    """Resume media playback if it was paused."""
    try:
        import subprocess
        # Try to resume Spotify
        spotify_script = '''
            tell application "Spotify"
                if player state is paused then
                    play
                    return true
                end if
            end tell
            return false
        '''
        result = subprocess.run(['osascript', '-e', spotify_script], 
                              capture_output=True, timeout=1)
        if result.returncode == 0 and b'true' in result.stdout:
            log_info("[Audio] Resumed Spotify")
            return True
        
        # Try to resume YouTube (Chrome/Safari)
        youtube_script = '''
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is true
                if frontApp is "Google Chrome" or frontApp is "Safari" then
                    keystroke "k" using {command down}
                    return true
                end if
            end tell
            return false
        '''
        result = subprocess.run(['osascript', '-e', youtube_script], 
                              capture_output=True, timeout=1)
        if result.returncode == 0:
            log_info("[Audio] Resumed media playback")
            return True
        
        # Try to resume Music app
        music_script = '''
            tell application "Music"
                if player state is paused then
                    play
                    return true
                end if
            end tell
            return false
        '''
        result = subprocess.run(['osascript', '-e', music_script], 
                              capture_output=True, timeout=1)
        if result.returncode == 0 and b'true' in result.stdout:
            log_info("[Audio] Resumed Music app")
            return True
        
        return False
    except Exception as e:
        log_info(f"[Audio] Could not resume media: {e}")
        return False

def start_recording():
    """Start recording with bulletproof error handling - NEVER crashes."""
    global is_recording, audio_stream, audio_frames, app_instance_global, selected_model_name
    
    # CRASH PREVENTION: Wrap everything in try-except
    try:
        # CRASH PREVENTION: Check if already recording
        if is_recording:
            print("[Warning] Already recording, ignoring request.")
            return
        
        # CRASH PREVENTION: Validate audio system first
        try:
            import sounddevice as sd
            # Test if audio system is available
            _ = sd.query_devices()
        except Exception as audio_error:
            print(f"[Error] Audio system unavailable: {audio_error}", file=sys.stderr)
            try:
                status_queue.put(f"{APP_ICON_ERROR} Audio Unavailable")
            except:
                pass
            return
        
        # CRASH PREVENTION: Check if cloud API is enabled (doesn't need local model)
        use_cloud = False
        try:
            from config import USE_CLOUD_API
            use_cloud = USE_CLOUD_API
        except Exception as config_error:
            print(f"[Warning] Could not read USE_CLOUD_API: {config_error}")
            use_cloud = False
        
        # CRASH PREVENTION: Check if model is available (either Faster Whisper or regular) or cloud API
        model_available = use_cloud  # Cloud API doesn't need local model
        
        if not model_available:
            try:
                if USE_FASTER_WHISPER and FASTER_WHISPER_AVAILABLE:
                    model_available = (faster_whisper_model is not None)
                else:
                    model_available = (transcription_pipe is not None)
            except Exception as model_check_error:
                print(f"[Warning] Model check failed: {model_check_error}")
                model_available = False
        
        # CRASH PREVENTION: Lazy loading with error handling
        if not model_available:
            try:
                from config import LAZY_MODEL_LOADING
                if LAZY_MODEL_LOADING and selected_model_name:
                    print("[Lazy Loading] Loading model on first use...")
                    try:
                        model_id = MODEL_MAP.get(selected_model_name)
                        if model_id:
                            # Load model synchronously (quick load)
                            load_result = load_model_and_processor(model_id, cache_only=False)
                            if load_result:
                                # Re-check availability
                                try:
                                    if USE_FASTER_WHISPER and FASTER_WHISPER_AVAILABLE:
                                        model_available = (faster_whisper_model is not None)
                                    else:
                                        model_available = (transcription_pipe is not None)
                                except:
                                    model_available = False
                    except Exception as load_error:
                        print(f"[Warning] Lazy loading failed: {load_error}")
                        model_available = False
            except Exception as lazy_error:
                print(f"[Warning] Lazy loading check failed: {lazy_error}")
        
        # CRASH PREVENTION: Validate model before proceeding
        if not model_available:
            print("[Error] Model not loaded, cannot start recording.")
            try:
                status_queue.put(f"{APP_ICON_ERROR} No Model Loaded")
            except:
                pass
            return

        # CRASH PREVENTION: Clean up any existing stream first (with multiple safety checks)
        if audio_stream is not None:
            try:
                try:
                    if hasattr(audio_stream, 'active') and audio_stream.active:
                        audio_stream.stop()
                except:
                    pass
                try:
                    if hasattr(audio_stream, 'close'):
                        audio_stream.close()
                except:
                    pass
            except:
                pass
            finally:
                audio_stream = None

        # CRASH PREVENTION: Auto-pause media if enabled (non-blocking)
        global media_was_paused
        media_was_paused = False
        try:
            from config import AUTO_PAUSE_MEDIA
            if AUTO_PAUSE_MEDIA:
                try:
                    media_was_paused = pause_media_playback()
                except:
                    media_was_paused = False
        except:
            pass
        
        # CRASH PREVENTION: Initialize state variables safely
        try:
            print("[Info] Starting recording...")
            try:
                status_queue.put(f"{APP_ICON_RECORDING} Recording...")
            except:
                pass
            audio_frames = []
            is_recording = True
        except Exception as state_error:
            print(f"[Error] Failed to initialize recording state: {state_error}", file=sys.stderr)
            is_recording = False
            return
        
        # CRASH PREVENTION: Update menu if app instance exists (non-blocking)
        try:
            if app_instance_global and hasattr(app_instance_global, 'update_record_menu'):
                try:
                    app_instance_global.update_record_menu()
                except:
                    pass
                if hasattr(app_instance_global, 'update_menu_states'):
                    try:
                        app_instance_global.update_menu_states()
                    except:
                        pass
        except:
            pass
        
        # CRASH PREVENTION: Start audio stream with comprehensive error handling
        stream_started = False
        try:
            # Get sample rate safely
            try:
                current_sample_rate = get_optimal_sample_rate()
                if not isinstance(current_sample_rate, (int, float)) or current_sample_rate <= 0:
                    current_sample_rate = SAMPLE_RATE  # Fallback to default
            except:
                current_sample_rate = SAMPLE_RATE  # Fallback to default
            
            # CRASH PREVENTION: Get selected audio input device (always use built-in microphone)
            input_device = None
            auto_detect = True
            try:
                from config import AUDIO_INPUT_DEVICE, AUTO_DETECT_BUILTIN_MIC
                input_device = AUDIO_INPUT_DEVICE
                auto_detect = AUTO_DETECT_BUILTIN_MIC
            except:
                input_device = None
                auto_detect = True  # Default to auto-detect
            
            # CRASH PREVENTION: Auto-detect built-in microphone with error handling
            if auto_detect and input_device is None:
                try:
                    # Try to get built-in microphone from app instance
                    if app_instance_global and hasattr(app_instance_global, 'get_builtin_microphone'):
                        try:
                            builtin_mic = app_instance_global.get_builtin_microphone()
                            if builtin_mic and isinstance(builtin_mic, dict) and 'index' in builtin_mic:
                                input_device = builtin_mic['index']
                                log_info(f"[Audio] Auto-detected built-in microphone: {builtin_mic.get('name', 'Unknown')} (input only)")
                        except:
                            pass
                    
                    # Fallback: find built-in microphone directly
                    if input_device is None:
                        try:
                            devices = sd.query_devices()
                            if devices and len(devices) > 0:
                                for i, device in enumerate(devices):
                                    try:
                                        if device.get('max_input_channels', 0) > 0:
                                            device_name = device.get('name', '').lower()
                                            # Look for built-in MacBook microphone
                                            if any(keyword in device_name for keyword in ['built-in', 'builtin', 'internal', 'macbook']):
                                                input_device = i
                                                log_info(f"[Audio] Found built-in microphone: {device.get('name', 'Unknown')} (input only)")
                                                break
                                    except:
                                        continue
                                
                                # If still not found, use default input device
                                if input_device is None:
                                    try:
                                        default_input = sd.default.device[0]
                                        if default_input is not None:
                                            input_device = default_input
                                            device_info = sd.query_devices(default_input)
                                            log_info(f"[Audio] Using default input device: {device_info.get('name', 'Unknown')} (input only)")
                                    except:
                                        pass
                        except Exception as device_error:
                            log_warning(f"[Audio] Could not detect built-in microphone: {device_error}")
                except:
                    pass
            
            # CRASH PREVENTION: Validate device before using
            if input_device is not None:
                try:
                    # Verify device exists and is valid
                    device_info = sd.query_devices(input_device)
                    if device_info.get('max_input_channels', 0) == 0:
                        print(f"[Warning] Device {input_device} has no input channels, using default")
                        input_device = None
                except:
                    print(f"[Warning] Device {input_device} invalid, using default")
                    input_device = None
            
            # CRASH PREVENTION: Create stream with selected device (microphone only, no system audio)
            stream_kwargs = {
                'samplerate': int(current_sample_rate),
                'channels': CHANNELS,
                'callback': audio_callback,
                'dtype': 'float32'
            }
            
            # Always use input device (microphone) - never system audio
            if input_device is not None:
                try:
                    stream_kwargs['device'] = int(input_device)
                    try:
                        device_info = sd.query_devices(input_device)
                        log_info(f"[Audio] Using microphone: {device_info.get('name', 'Unknown')} (input only)")
                    except:
                        log_info(f"[Audio] Using device: {input_device}")
                except:
                    log_info("[Audio] Using default input device (built-in microphone)")
            
            # CRASH PREVENTION: Create and start stream with error handling
            try:
                audio_stream = sd.InputStream(**stream_kwargs)
                if audio_stream is not None:
                    try:
                        audio_stream.start()
                        stream_started = True
                        print(f"[Info] Audio stream started at {current_sample_rate}Hz")
                    except Exception as start_error:
                        print(f"[Error] Failed to start audio stream: {start_error}", file=sys.stderr)
                        try:
                            audio_stream.close()
                        except:
                            pass
                        audio_stream = None
                        stream_started = False
            except Exception as stream_error:
                print(f"[Error] Failed to create audio stream: {stream_error}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                audio_stream = None
                stream_started = False
            
            # CRASH PREVENTION: If stream failed, reset state
            if not stream_started:
                is_recording = False
                audio_stream = None
                try:
                    status_queue.put(f"{APP_ICON_ERROR} Mic Error")
                except:
                    pass
                # Update menu to reflect failure
                try:
                    if app_instance_global and hasattr(app_instance_global, 'update_record_menu'):
                        app_instance_global.update_record_menu()
                    if app_instance_global and hasattr(app_instance_global, 'update_menu_states'):
                        app_instance_global.update_menu_states()
                except:
                    pass
                return
                
        except Exception as stream_error:
            print(f"[CRITICAL] Audio stream setup failed: {stream_error}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            # CRASH PREVENTION: Always reset state on error
            is_recording = False
            audio_stream = None
            try:
                status_queue.put(f"{APP_ICON_ERROR} Mic Error")
            except:
                pass
            # Update menu to reflect failure
            try:
                if app_instance_global and hasattr(app_instance_global, 'update_record_menu'):
                    app_instance_global.update_record_menu()
                if app_instance_global and hasattr(app_instance_global, 'update_menu_states'):
                    app_instance_global.update_menu_states()
            except:
                pass
            return
                
    except Exception as e:
        # CRASH PREVENTION: Final safety net - catch ANY unexpected error
        print(f"[CRITICAL] Unexpected error in start_recording: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # CRASH PREVENTION: Always reset to safe state
        is_recording = False
        try:
            if audio_stream is not None:
                try:
                    if hasattr(audio_stream, 'active') and audio_stream.active:
                        audio_stream.stop()
                except:
                    pass
                try:
                    audio_stream.close()
                except:
                    pass
        except:
            pass
        audio_stream = None
        try:
            status_queue.put(f"{APP_ICON_ERROR} Start Failed")
        except:
            pass
        # Update menu to reflect failure
        try:
            if app_instance_global and hasattr(app_instance_global, 'update_record_menu'):
                app_instance_global.update_record_menu()
            if app_instance_global and hasattr(app_instance_global, 'update_menu_states'):
                app_instance_global.update_menu_states()
        except:
            pass

def stop_recording_and_transcribe():
    """Safely stop recording and start transcription with comprehensive error handling."""
    global is_recording, audio_stream, audio_frames, is_processing, app_instance_global, media_was_paused
    
    try:
        if not is_recording:
            print("[Info] Not recording, nothing to stop.")
            return

        print("[Info] Recording stopped.")
        is_recording = False  # Set this FIRST to prevent callback issues
        
        # Safely stop audio stream
        if audio_stream is not None:
            try:
                if audio_stream.active:
                    audio_stream.stop()
                audio_stream.close()
            except Exception as e:
                print(f"[Warning] Error stopping audio stream: {e}", file=sys.stderr)
            finally:
                audio_stream = None

        # Update menu safely
        try:
            if app_instance_global and hasattr(app_instance_global, 'update_record_menu'):
                app_instance_global.update_record_menu()
                if hasattr(app_instance_global, 'update_menu_states'):
                    app_instance_global.update_menu_states()
        except Exception as e:
            print(f"[Warning] Failed to update menu: {e}", file=sys.stderr)
        
        # Check if we have audio data
        if not audio_frames or len(audio_frames) == 0:
            print("[Warning] No audio frames captured.")
            status_queue.put(f"{APP_ICON_IDLE} Ready ({selected_model_name})")
            audio_frames = []
            return
        
        # Check for empty frames
        try:
            if all(frame.size == 0 for frame in audio_frames):
                print("[Warning] Audio frames captured, but they are all empty.")
                status_queue.put(f"{APP_ICON_IDLE} Ready ({selected_model_name})")
                audio_frames = []
                return
        except Exception as e:
            print(f"[Warning] Error checking audio frames: {e}", file=sys.stderr)
            audio_frames = []
            status_queue.put(f"{APP_ICON_IDLE} Ready ({selected_model_name})")
            return

        # Process audio data safely
        try:
            audio_data = np.concatenate(audio_frames, axis=0).astype(np.float32)
            print(f"[Debug] Concatenated audio data shape: {audio_data.shape}")
            
            # Validate audio data
            if len(audio_data) == 0:
                print("[Warning] Audio data is empty after concatenation.")
                status_queue.put(f"{APP_ICON_IDLE} Ready ({selected_model_name})")
                audio_frames = []
                return

        except Exception as e:
            print(f"[Error] Failed to process audio frames: {e}", file=sys.stderr)
            status_queue.put(f"{APP_ICON_ERROR} Audio Error")
            audio_frames = []
            return

        # Clear frames before starting transcription
        audio_frames = []
        
        # Resume media if it was paused and option is enabled
        try:
            from config import RESUME_MEDIA_AFTER_RECORDING
            if RESUME_MEDIA_AFTER_RECORDING and media_was_paused:
                resume_media_playback()
                media_was_paused = False
        except:
            pass
        
        # Check if model is available (either Faster Whisper or regular)
        model_available = False
        if USE_FASTER_WHISPER and FASTER_WHISPER_AVAILABLE:
            model_available = (faster_whisper_model is not None)
        else:
            model_available = (transcription_pipe is not None)
        
        if not model_available:
            print("[Error] Model not loaded, cannot transcribe.")
            status_queue.put(f"{APP_ICON_ERROR} No Model")
            return
        
        # Start transcription safely
        try:
            cancel_processing_event.clear()
            is_processing = True
            status_queue.put(f"{APP_ICON_PROCESSING} Processing audio...")
            print("[Info] Launching transcription thread...")
            thread = threading.Thread(target=transcribe_audio_thread, args=(audio_data,), daemon=True)
            thread.start()
        except Exception as e:
            print(f"[Error] Failed to start transcription thread: {e}", file=sys.stderr)
            is_processing = False
            status_queue.put(f"{APP_ICON_ERROR} Thread Error")
            
    except Exception as e:
        print(f"[CRITICAL] Unexpected error in stop_recording_and_transcribe: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Reset everything to safe state
        is_recording = False
        is_processing = False
        audio_frames = []
        if audio_stream:
            try:
                audio_stream.stop()
                audio_stream.close()
            except:
                pass
            audio_stream = None
        status_queue.put(f"{APP_ICON_ERROR} Crash Prevented")


# --- Keyboard Listener Callbacks ---
def on_press(key):
    global trigger_key_held, is_processing, is_recording, audio_stream, audio_frames, shortcut_manager, app_enabled, TRIGGER_KEY, TRIGGER_KEY_2
    
    # Handle shortcuts via manager
    if shortcut_manager:
        shortcut_manager.handle_key_press(key)
    
    # Check if app is enabled
    if not app_enabled:
        return
    
    # Handle trigger keys (push-to-talk) - support both primary and secondary
    trigger_key = TRIGGER_KEY
    trigger_key_2 = TRIGGER_KEY_2
    
    # Load from config if not set
    if trigger_key is None:
        try:
            from config import TRIGGER_KEY as config_trigger
            trigger_key = parse_key_shortcut(config_trigger)
            TRIGGER_KEY = trigger_key
        except:
            trigger_key = keyboard.Key.alt_r  # Fallback
    
    if trigger_key_2 is None:
        try:
            from config import TRIGGER_KEY_2 as config_trigger_2
            if config_trigger_2 and config_trigger_2.lower() != 'none':
                trigger_key_2 = parse_key_shortcut(config_trigger_2)
                TRIGGER_KEY_2 = trigger_key_2
        except:
            trigger_key_2 = None  # Secondary key is optional
    
    # Check if this is either trigger key
    is_trigger = False
    
    # Check primary trigger key
    if isinstance(trigger_key, keyboard.Key):
        is_trigger = (key == trigger_key)
    elif isinstance(trigger_key, str):
        try:
            key_char = key.char if hasattr(key, 'char') and key.char else None
            is_trigger = (key_char == trigger_key.lower())
        except:
            is_trigger = False
    
    # Check secondary trigger key if primary didn't match
    if not is_trigger and trigger_key_2:
        if isinstance(trigger_key_2, keyboard.Key):
            is_trigger = (key == trigger_key_2)
        elif isinstance(trigger_key_2, str):
            try:
                key_char = key.char if hasattr(key, 'char') and key.char else None
                is_trigger = (key_char == trigger_key_2.lower())
            except:
                is_trigger = False
        # Special handling for Fn key - try to detect it by key code
        if not is_trigger and trigger_key_2 in [getattr(keyboard.Key, 'fn', None), getattr(keyboard.Key, 'f24', None)]:
            # Try to detect Fn key by checking if it's a special key
            # On some keyboards, Fn sends a special keycode
            try:
                # Check if key has a special attribute that might indicate Fn
                key_name = str(key).lower()
                if 'fn' in key_name or 'function' in key_name:
                    is_trigger = True
            except:
                pass
    
    if is_trigger:
        if not trigger_key_held:
            trigger_key_held = True
            print("[Input] Trigger key pressed.")

            # --- IMMEDIATE CANCELLATION AND RESET ---
            # Force immediate stop of any processing
            if is_processing:
                print("[Info] FORCE STOPPING processing and resetting...")
                cancel_processing_event.set()
                is_processing = False  # Force reset immediately
                
            # Force stop any active recording immediately
            if is_recording:
                print("[Info] FORCE STOPPING active recording...")
                is_recording = False
                if audio_stream:
                    try:
                        audio_stream.stop()
                        audio_stream.close()
                    except Exception as e:
                        print(f"[Error] Failed to stop/close stream: {e}", file=sys.stderr)
                    finally:
                        audio_stream = None
                audio_frames = []

            # Reset status immediately
            status_queue.put(f"{APP_ICON_RECORDING} Recording...")
            
            # Clear any pending items in queues
            try:
                while not status_queue.empty():
                    status_queue.get_nowait()
                    status_queue.task_done()
            except:
                pass
                
            try:
                while not transcription_queue.empty():
                    transcription_queue.get_nowait()
                    transcription_queue.task_done()
            except:
                pass

            # Start fresh recording
            print("[Info] Starting FRESH recording...")
            start_recording()

def on_release(key):
    global trigger_key_held, is_recording, TRIGGER_KEY, TRIGGER_KEY_2
    
    # Check if released key matches either trigger key
    is_trigger_release = False
    
    # Check primary trigger key
    if key == TRIGGER_KEY:
        is_trigger_release = True
    # Check secondary trigger key
    elif TRIGGER_KEY_2 and key == TRIGGER_KEY_2:
        is_trigger_release = True
    # Special handling for Fn key detection
    elif TRIGGER_KEY_2:
        try:
            key_name = str(key).lower()
            if 'fn' in key_name or 'function' in key_name:
                is_trigger_release = True
        except:
            pass
    
    if is_trigger_release:
        if trigger_key_held:
            trigger_key_held = False
            print("[Input] Trigger key released.")

            # Simple logic: if recording, stop and transcribe
            if is_recording:
                print("[Info] Stopping recording and starting transcription...")
                stop_recording_and_transcribe()
            else:
                print("[Info] Trigger key released, but no active recording found.")
                # Reset to idle state
                status_queue.put(f"{APP_ICON_IDLE} Ready ({selected_model_name})")

def start_keyboard_listener():
    """Starts the keyboard listener in a separate thread."""
    global keyboard_listener, listener_thread
    if listener_thread is None or not listener_thread.is_alive():
        try:
            keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener_thread = threading.Thread(target=keyboard_listener.run, daemon=True)
            listener_thread.start()
            print("Keyboard listener started.")
        except Exception as e:
            print(f"Failed to start keyboard listener: {e}", file=sys.stderr)
            # Attempt to stop if partially started
            if keyboard_listener and hasattr(keyboard_listener, 'stop'):
                keyboard_listener.stop()
            keyboard_listener = None
            listener_thread = None
            status_queue.put(f"{APP_ICON_ERROR} Listener Failed")
    else:
        print("Keyboard listener already running.")


# --- Rumps Application Class ---
class WhisperStatusBarApp(rumps.App):
    def __init__(self):
        global app_instance_global, selected_model_name, DEFAULT_MODEL_NAME, shortcut_manager, TRIGGER_KEY, TRIGGER_KEY_2
        
        super(WhisperStatusBarApp, self).__init__(APP_NAME, title=APP_ICON_IDLE, quit_button='Quit')
        app_instance_global = self  # Store global reference for menu updates
        
        # Initialize keyboard shortcut manager
        shortcut_manager = KeyboardShortcutManager()
        
        # Load trigger keys from config
        try:
            from config import TRIGGER_KEY as config_trigger
            TRIGGER_KEY = parse_key_shortcut(config_trigger)
        except:
            TRIGGER_KEY = keyboard.Key.alt_r  # Fallback
        
        try:
            from config import TRIGGER_KEY_2 as config_trigger_2
            if config_trigger_2 and config_trigger_2.lower() != 'none':
                TRIGGER_KEY_2 = parse_key_shortcut(config_trigger_2)
                if TRIGGER_KEY_2:
                    print(f"[Shortcuts] Secondary trigger key loaded: {config_trigger_2} → {TRIGGER_KEY_2}")
                else:
                    print(f"[Shortcuts] Warning: Could not parse secondary trigger key '{config_trigger_2}'. It may not be supported on your keyboard.")
                    TRIGGER_KEY_2 = None
            else:
                TRIGGER_KEY_2 = None
        except Exception as e:
            print(f"[Shortcuts] Error loading secondary trigger key: {e}")
            TRIGGER_KEY_2 = None  # Secondary key is optional
        
        # Auto-detect best model if not set
        if DEFAULT_MODEL_NAME is None:
            optimal_model = detect_optimal_model()
            DEFAULT_MODEL_NAME = optimal_model
            selected_model_name = optimal_model
            print(f"[Init] Auto-selected model: {optimal_model}")
        else:
            selected_model_name = DEFAULT_MODEL_NAME
        
        # Schedule dock hiding for after app initialization
        try:
            from config import HIDE_FROM_DOCK
            if HIDE_FROM_DOCK:
                threading.Timer(0.5, self.configure_dock_visibility).start()
            else:
                print("[Info] App will appear in dock (HIDE_FROM_DOCK=False)")
        except Exception as e:
            print(f"[Warning] Could not configure dock visibility: {e}")
        
        # Direct Start/Stop buttons in menu bar
        self.start_menu = rumps.MenuItem("▶️ Start", callback=self.start_recording_menu)
        self.stop_menu = rumps.MenuItem("⏹️ Stop", callback=self.stop_recording_menu)
        
        self.model_menu = rumps.MenuItem("Select Model")
        self.mode_menu = rumps.MenuItem("Processing Mode")
        self.audio_menu = rumps.MenuItem("🎤 Audio Settings")
        self.faster_whisper_menu = rumps.MenuItem("⚡ Faster Whisper", callback=self.toggle_faster_whisper)
        self.record_menu = rumps.MenuItem("🎤 Start Recording (Default)", callback=self.toggle_recording)
        
        # Add keyboard shortcuts menu
        self.shortcuts_menu = rumps.MenuItem("Keyboard Shortcuts")
        self.menu = [
            self.start_menu,
            self.stop_menu,
            None,  # Separator
            self.record_menu,
            None,  # Separator
            rumps.MenuItem("Keyboard: Hold Option Key", callback=None),  # Info item
            None,  # Separator
            self.model_menu, 
            self.mode_menu,
            self.audio_menu,
            self.faster_whisper_menu,
            self.shortcuts_menu,
            None  # Separator
        ]
        self.create_model_submenu()
        self.create_mode_submenu()
        self.create_audio_submenu()
        self.create_shortcuts_submenu()
        self.load_thread = None

        # Configure queue check interval based on low power mode
        try:
            from config import ENABLE_LOW_POWER_MODE, IDLE_QUEUE_CHECK_INTERVAL
            check_interval = IDLE_QUEUE_CHECK_INTERVAL if ENABLE_LOW_POWER_MODE else 1.0
        except:
            check_interval = 1.0

        # Start background thread to check queues
        self.queue_timer = rumps.Timer(self.check_queues, check_interval)
        self.queue_timer.start()

        # Update Faster Whisper menu state
        self.faster_whisper_menu.state = USE_FASTER_WHISPER
        
        # Load model on startup (respect lazy loading setting)
        try:
            from config import LAZY_MODEL_LOADING, USE_CLOUD_API
            # If cloud API is enabled, no need to load local model
            if USE_CLOUD_API:
                print("[Init] Cloud API enabled, skipping local model load")
                status_queue.put(f"{APP_ICON_SUCCESS} Ready • Cloud API")
            elif not LAZY_MODEL_LOADING:
                # Load model immediately in background
                print(f"[Init] Loading model: {selected_model_name}")
                self.change_model(None, model_name=selected_model_name)
            else:
                print(f"[Init] Lazy loading enabled - model will load on first use")
                status_queue.put(f"{APP_ICON_IDLE} Ready • Lazy Load")
        except Exception as e:
            # Default: load model immediately
            print(f"[Init] Loading model (fallback): {selected_model_name}")
            print(f"[Init] Error checking config: {e}")
            self.change_model(None, model_name=selected_model_name)
        
        # Initialize VAD model in background (only if needed)
        try:
            from config import DISABLE_BACKGROUND_PRELOADING
            if not DISABLE_BACKGROUND_PRELOADING:
                threading.Timer(2.0, load_vad_model).start()
        except:
            threading.Timer(2.0, load_vad_model).start()

        # Start keyboard listener after a short delay
        threading.Timer(1.0, start_keyboard_listener).start()

        # Update record menu item text based on state
        self.update_record_menu()
        # Update Start/Stop menu states
        self.update_menu_states()

    def toggle_faster_whisper(self, sender):
        """Toggle Faster Whisper on/off."""
        global USE_FASTER_WHISPER, faster_whisper_model, transcription_pipe, current_model_id
        
        USE_FASTER_WHISPER = not USE_FASTER_WHISPER
        sender.state = USE_FASTER_WHISPER
        
        if USE_FASTER_WHISPER:
            print("[Info] ⚡ Faster Whisper enabled (2-4x faster)")
            status_queue.put(f"{APP_ICON_LOADING} Switching to Faster Whisper...")
        else:
            print("[Info] Using regular Whisper")
            status_queue.put(f"{APP_ICON_LOADING} Switching to regular Whisper...")
        
        # Reload current model with new engine
        if current_model_id:
            model_name = selected_model_name
            threading.Thread(target=lambda: self.change_model(None, model_name=model_name), daemon=True).start()

    def create_model_submenu(self):
        """Creates or updates the model selection submenu with grouping and indicators."""
        # Check if the underlying NSMenu exists before trying to clear
        if self.model_menu._menu is not None:
            self.model_menu.clear()
        else:
             print("Skipping initial model_menu.clear() as _menu is None.")

        # Clear existing dictionary items manually just in case clear() didn't run
        existing_keys = list(self.model_menu.keys())
        for key in existing_keys:
             if key in self.model_menu:
                 del self.model_menu[key]

        # Add "Auto-Select Best" option at top
        auto_select_item = rumps.MenuItem("🎯 Auto-Select Best", callback=self.auto_select_model)
        auto_select_item.state = False  # Not a toggle, just an action
        self.model_menu["🎯 Auto-Select Best"] = auto_select_item
        
        self.model_menu[None] = None  # Separator
        
        # Add Local Models section
        local_header = rumps.MenuItem("🏠 Local Models", callback=None)
        local_header.set_callback(None)  # Non-clickable header
        self.model_menu["🏠 Local Models"] = local_header
        
        # Add local models with metadata
        for name in MODEL_MAP.keys():
            if name in MODEL_METADATA:
                meta = MODEL_METADATA[name]
                display_name = f"{name} {meta.get('speed', '')} {meta.get('accuracy', '')} ({meta.get('size', '')})"
            else:
                display_name = name
            
            callback_func = lambda sender, captured_name=name: self.change_model(sender, model_name=captured_name)
            item = rumps.MenuItem(display_name, callback=callback_func)
            item.state = (name == selected_model_name)
            self.model_menu[display_name] = item
        
        # Add Cloud APIs section
        try:
            from config import USE_CLOUD_API
            if USE_CLOUD_API:
                self.model_menu[None] = None  # Separator
                cloud_header = rumps.MenuItem("☁️ Cloud APIs", callback=None)
                cloud_header.set_callback(None)
                self.model_menu["☁️ Cloud APIs"] = cloud_header
                
                # Add cloud API options
                cloud_providers = ["OpenAI", "Google", "Deepgram", "Custom"]
                for provider in cloud_providers:
                    callback_func = lambda sender, p=provider: self.select_cloud_provider(sender, p)
                    item = rumps.MenuItem(f"☁️ {provider}", callback=callback_func)
                    try:
                        from config import CLOUD_PROVIDER
                        item.state = (CLOUD_PROVIDER.lower() == provider.lower())
                    except:
                        item.state = False
                    self.model_menu[f"☁️ {provider}"] = item
        except:
            pass

    def create_mode_submenu(self):
        """Creates the processing mode selection submenu."""
        try:
            from config import PROCESSING_MODE
            
            # Traditional Mode
            traditional_item = rumps.MenuItem(
                "Traditional Processing", 
                callback=self.set_traditional_mode
            )
            traditional_item.state = (PROCESSING_MODE == "Traditional")
            self.mode_menu["Traditional Processing"] = traditional_item
            
            # Optimized Mode
            optimized_item = rumps.MenuItem(
                "Optimized Processing", 
                callback=self.set_optimized_mode
            )
            optimized_item.state = (PROCESSING_MODE == "Optimized")
            self.mode_menu["Optimized Processing"] = optimized_item
            
            # Ultra-Fast Mode
            ultra_fast_item = rumps.MenuItem(
                "Ultra-Fast Processing", 
                callback=self.set_ultra_fast_mode
            )
            ultra_fast_item.state = (PROCESSING_MODE == "Ultra-Fast")
            self.mode_menu["Ultra-Fast Processing"] = ultra_fast_item
            
        except Exception as e:
            print(f"[Warning] Failed to create mode submenu: {e}")
    
    def create_audio_submenu(self):
        """Creates the audio settings submenu with microphone selection and options."""
        try:
            # Get available microphones (input devices only, no system audio)
            input_devices = self.get_audio_input_devices()
            
            # Get built-in microphone
            builtin_mic = self.get_builtin_microphone()
            
            # Add microphone selection header
            mic_header = rumps.MenuItem("🎤 Select Microphone", callback=None)
            self.audio_menu["🎤 Select Microphone"] = mic_header
            
            # Add each microphone as an option
            try:
                from config import AUDIO_INPUT_DEVICE, AUTO_DETECT_BUILTIN_MIC
                current_device = AUDIO_INPUT_DEVICE
                auto_detect = AUTO_DETECT_BUILTIN_MIC
            except:
                current_device = None
                auto_detect = True
            
            # Add "Built-in Microphone (Auto)" option - recommended
            if builtin_mic:
                is_builtin_selected = (auto_detect and current_device is None) or current_device == builtin_mic['index']
                builtin_item = rumps.MenuItem(
                    f"🎤 Built-in Mic ({builtin_mic['name']})", 
                    callback=lambda _: self.select_microphone(builtin_mic['index']),
                    state=1 if is_builtin_selected else 0
                )
                self.audio_menu[f"🎤 Built-in Mic ({builtin_mic['name']})"] = builtin_item
            
            # Add "Auto-Detect Built-in" option
            auto_detect_state = 1 if (auto_detect and current_device is None) else 0
            auto_item = rumps.MenuItem(
                "🔍 Auto-Detect Built-in", 
                callback=lambda _: self.select_microphone(None),
                state=auto_detect_state
            )
            self.audio_menu["🔍 Auto-Detect Built-in"] = auto_item
            
            # Add separator before other microphones
            self.audio_menu[None] = None
            
            # Add other available microphones (excluding built-in which is already shown)
            for device_info in input_devices:
                device_id = device_info['index']
                device_name = device_info['name']
                
                # Skip built-in microphone (already shown at top)
                if builtin_mic and device_id == builtin_mic['index']:
                    continue
                
                is_selected = (current_device == device_id or 
                              (isinstance(current_device, str) and current_device == device_name))
                
                item = rumps.MenuItem(
                    f"🎤 {device_name}",
                    callback=lambda _, d=device_id: self.select_microphone(d),
                    state=1 if is_selected else 0
                )
                self.audio_menu[f"🎤 {device_name}"] = item
            
            # Add separator
            self.audio_menu[None] = None
            
            # Add audio handling options
            try:
                from config import AUTO_PAUSE_MEDIA, RESUME_MEDIA_AFTER_RECORDING, SHARE_MIC_DURING_CALLS
                
                pause_media_item = rumps.MenuItem(
                    "⏸️ Auto-Pause Media",
                    callback=self.toggle_auto_pause_media,
                    state=1 if AUTO_PAUSE_MEDIA else 0
                )
                self.audio_menu["⏸️ Auto-Pause Media"] = pause_media_item
                
                resume_media_item = rumps.MenuItem(
                    "▶️ Resume Media After Recording",
                    callback=self.toggle_resume_media,
                    state=1 if RESUME_MEDIA_AFTER_RECORDING else 0
                )
                self.audio_menu["▶️ Resume Media After Recording"] = resume_media_item
                
                share_mic_item = rumps.MenuItem(
                    "📞 Share Mic During Calls",
                    callback=self.toggle_share_mic_calls,
                    state=1 if SHARE_MIC_DURING_CALLS else 0
                )
                self.audio_menu["📞 Share Mic During Calls"] = share_mic_item
            except Exception as e:
                log_warning(f"Failed to load audio options: {e}")
                
        except Exception as e:
            log_warning(f"Failed to create audio submenu: {e}")
    
    def get_audio_input_devices(self):
        """Get list of available audio input devices (microphones only, no system audio)."""
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                # Only include devices that have input channels (microphones)
                # Exclude devices that are output-only or system audio
                if device['max_input_channels'] > 0:
                    device_name = device['name'].lower()
                    # Skip system audio/output devices
                    if 'system' not in device_name and 'output' not in device_name:
                        input_devices.append({
                            'index': i,
                            'name': device['name'],
                            'channels': device['max_input_channels'],
                            'sample_rate': int(device['default_samplerate']),
                            'is_builtin': self.is_builtin_microphone(device['name'])
                        })
            return input_devices
        except Exception as e:
            log_warning(f"Failed to get audio devices: {e}")
            return []
    
    def is_builtin_microphone(self, device_name):
        """Check if device is the built-in MacBook microphone."""
        name_lower = device_name.lower()
        # Common names for built-in MacBook microphones
        builtin_keywords = [
            'built-in',
            'builtin',
            'internal',
            'macbook',
            'macbook pro',
            'macbook air',
            'imac',
            'default input'
        ]
        return any(keyword in name_lower for keyword in builtin_keywords)
    
    def get_builtin_microphone(self):
        """Get the built-in MacBook microphone device."""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name = device['name']
                    if self.is_builtin_microphone(device_name):
                        return {
                            'index': i,
                            'name': device_name,
                            'channels': device['max_input_channels'],
                            'sample_rate': int(device['default_samplerate'])
                        }
            # Fallback: use default input device
            default_input = sd.default.device[0]
            if default_input is not None:
                default_device = devices[default_input]
                return {
                    'index': default_input,
                    'name': default_device['name'],
                    'channels': default_device['max_input_channels'],
                    'sample_rate': int(default_device['default_samplerate'])
                }
            return None
        except Exception as e:
            log_warning(f"Failed to get built-in microphone: {e}")
            return None
    
    def select_microphone(self, device_id):
        """Select microphone device (always microphone input, never system audio)."""
        try:
            if device_id is None:
                # Auto-detect built-in microphone
                device_name = "Auto-Detect Built-in"
                update_config_value('AUDIO_INPUT_DEVICE', 'None')
                update_config_value('AUTO_DETECT_BUILTIN_MIC', 'True')
                builtin_mic = self.get_builtin_microphone()
                if builtin_mic:
                    device_name = f"Built-in: {builtin_mic['name']}"
            else:
                # Verify it's an input device (microphone), not system audio
                devices = sd.query_devices()
                if device_id < len(devices):
                    device_info = devices[device_id]
                    if device_info['max_input_channels'] == 0:
                        log_warning(f"Device {device_id} is not an input device (microphone)")
                        rumps.notification(
                            title="Invalid Device",
                            subtitle="Selected device is not a microphone",
                            message="Please select a microphone input device"
                        )
                        return
                    device_name = device_info['name']
                    update_config_value('AUDIO_INPUT_DEVICE', str(device_id))
                    update_config_value('AUTO_DETECT_BUILTIN_MIC', 'False')
                else:
                    log_warning(f"Device {device_id} not found")
                    return
            
            log_info(f"[Audio] Selected microphone: {device_name} (input only, no system audio)")
            rumps.notification(
                title="Microphone Selected",
                subtitle=f"Using: {device_name}",
                message="Microphone input only - no system audio"
            )
            # Recreate menu to update checkmarks
            self.create_audio_submenu()
        except Exception as e:
            log_error(f"Failed to select microphone: {e}")
    
    def toggle_auto_pause_media(self, sender):
        """Toggle auto-pause media option."""
        try:
            from config import AUTO_PAUSE_MEDIA
            new_value = not AUTO_PAUSE_MEDIA
            update_config_value('AUTO_PAUSE_MEDIA', str(new_value))
            sender.state = 1 if new_value else 0
            log_info(f"[Audio] Auto-pause media: {'Enabled' if new_value else 'Disabled'}")
        except Exception as e:
            log_error(f"Failed to toggle auto-pause media: {e}")
    
    def toggle_resume_media(self, sender):
        """Toggle resume media after recording option."""
        try:
            from config import RESUME_MEDIA_AFTER_RECORDING
            new_value = not RESUME_MEDIA_AFTER_RECORDING
            update_config_value('RESUME_MEDIA_AFTER_RECORDING', str(new_value))
            sender.state = 1 if new_value else 0
            log_info(f"[Audio] Resume media after recording: {'Enabled' if new_value else 'Disabled'}")
        except Exception as e:
            log_error(f"Failed to toggle resume media: {e}")
    
    def toggle_share_mic_calls(self, sender):
        """Toggle share mic during calls option."""
        try:
            from config import SHARE_MIC_DURING_CALLS
            new_value = not SHARE_MIC_DURING_CALLS
            update_config_value('SHARE_MIC_DURING_CALLS', str(new_value))
            sender.state = 1 if new_value else 0
            log_info(f"[Audio] Share mic during calls: {'Enabled' if new_value else 'Disabled'}")
        except Exception as e:
            log_error(f"Failed to toggle share mic calls: {e}")

    def set_traditional_mode(self, _):
        """Set Traditional processing mode."""
        try:
            update_config_value('PROCESSING_MODE', '"Traditional"')
            print("[Info] Switched to Traditional Processing (proven baseline VAD)")
            self.update_mode_menu_states()
        except Exception as e:
            print(f"[Error] Failed to set traditional mode: {e}")

    def set_optimized_mode(self, _):
        """Set Optimized processing mode."""
        try:
            update_config_value('PROCESSING_MODE', '"Optimized"')
            print("[Info] Switched to Optimized Processing (smart segmentation + your model)")
            self.update_mode_menu_states()
        except Exception as e:
            print(f"[Error] Failed to set optimized mode: {e}")

    def set_ultra_fast_mode(self, _):
        """Set Ultra-Fast processing mode."""
        try:
            update_config_value('PROCESSING_MODE', '"Ultra-Fast"')
            print("[Info] Switched to Ultra-Fast Processing (Tiny model override + optimization)")
            
            # Force reload current model to apply Ultra-Fast override
            print("[Info] Reloading model to apply Tiny model override...")
            self.change_model(None, model_name=selected_model_name)
            
            self.update_mode_menu_states()
        except Exception as e:
            print(f"[Error] Failed to set ultra-fast mode: {e}")

    def start_recording_menu(self, _):
        """Start recording directly from menu bar."""
        global is_recording
        if not is_recording:
            log_info("[Menu] Starting recording from menu bar...")
            start_recording()
            self.update_menu_states()
        else:
            log_info("[Menu] Already recording")
    
    def stop_recording_menu(self, _):
        """Stop recording directly from menu bar."""
        global is_recording
        if is_recording:
            log_info("[Menu] Stopping recording from menu bar...")
            stop_recording_and_transcribe()
            self.update_menu_states()
        else:
            log_info("[Menu] Not currently recording")
    
    def update_record_menu(self):
        """Update the record menu item text based on recording state."""
        global is_recording
        try:
            if is_recording:
                self.record_menu.title = "⏹ Stop Recording (Click to Stop)"
            else:
                self.record_menu.title = "🎤 Start Recording (Click Here or Hold Option)"
        except Exception as e:
            print(f"[Warning] Failed to update record menu: {e}")
    
    def update_menu_states(self):
        """Update Start/Stop menu item states based on recording state."""
        global is_recording
        try:
            if is_recording:
                # Recording: Start is disabled, Stop is enabled
                self.start_menu.set_callback(None)
                self.start_menu.title = "▶️ Start (Recording...)"
                self.stop_menu.set_callback(self.stop_recording_menu)
                self.stop_menu.title = "⏹️ Stop"
            else:
                # Not recording: Start is enabled, Stop is disabled
                self.start_menu.set_callback(self.start_recording_menu)
                self.start_menu.title = "▶️ Start"
                self.stop_menu.set_callback(None)
                self.stop_menu.title = "⏹️ Stop"
            # Also update the record menu
            self.update_record_menu()
        except Exception as e:
            log_warning(f"Failed to update menu states: {e}")

    def update_mode_menu_states(self):
        """Update the checkmark states in the mode menu."""
        try:
            from config import PROCESSING_MODE
            
            # Update menu item states
            if "Traditional Processing" in self.mode_menu:
                self.mode_menu["Traditional Processing"].state = (PROCESSING_MODE == "Traditional")
            if "Optimized Processing" in self.mode_menu:
                self.mode_menu["Optimized Processing"].state = (PROCESSING_MODE == "Optimized")
            if "Ultra-Fast Processing" in self.mode_menu:
                self.mode_menu["Ultra-Fast Processing"].state = (PROCESSING_MODE == "Ultra-Fast")
                
        except Exception as e:
            print(f"[Warning] Failed to update mode menu states: {e}")

    def toggle_recording(self, _):
        """Toggle recording via menu click - DEFAULT METHOD - with comprehensive error handling."""
        global is_recording, trigger_key_held
        try:
            if is_recording:
                # Stop recording
                print("[Menu] Stopping recording via menu (default method)...")
                trigger_key_held = False
                try:
                    stop_recording_and_transcribe()
                except Exception as stop_error:
                    print(f"[Error] stop_recording_and_transcribe() failed: {stop_error}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
                
                # Update menu with error handling
                try:
                    self.update_record_menu()
                except Exception as menu_error:
                    print(f"[Warning] Failed to update record menu: {menu_error}", file=sys.stderr)
                
                try:
                    self.update_menu_states()
                except Exception as state_error:
                    print(f"[Warning] Failed to update menu states: {state_error}", file=sys.stderr)
            else:
                # Start recording - this is the default/easiest method
                print("[Menu] Starting recording via menu (default method)...")
                try:
                    trigger_key_held = True
                    # Start recording with error handling
                    try:
                        start_recording()
                    except Exception as start_error:
                        print(f"[Error] start_recording() failed: {start_error}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                        raise start_error
                    
                    # Update menu with error handling
                    try:
                        self.update_record_menu()
                    except Exception as menu_error:
                        print(f"[Warning] Failed to update record menu: {menu_error}", file=sys.stderr)
                    
                    try:
                        self.update_menu_states()
                    except Exception as state_error:
                        print(f"[Warning] Failed to update menu states: {state_error}", file=sys.stderr)
                    
                    # Auto-stop after 15 seconds as safety timeout with error handling
                    def safe_auto_stop():
                        try:
                            global is_recording
                            if is_recording:
                                print("[Auto] Auto-stopping recording after 15 seconds...")
                                stop_recording_and_transcribe()
                            if hasattr(self, 'update_record_menu'):
                                try:
                                    self.update_record_menu()
                                except:
                                    pass
                                if hasattr(self, 'update_menu_states'):
                                    try:
                                        self.update_menu_states()
                                    except:
                                        pass
                        except Exception as e:
                            print(f"[Warning] Error in auto-stop: {e}", file=sys.stderr)
                    
                    try:
                        threading.Timer(15.0, safe_auto_stop).start()
                    except Exception as timer_error:
                        print(f"[Warning] Failed to start auto-stop timer: {timer_error}", file=sys.stderr)
                        
                except Exception as e:
                    print(f"[Error] Failed to start recording: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
                    try:
                        status_queue.put(f"{APP_ICON_ERROR} Start Failed")
                    except:
                        pass
                    trigger_key_held = False
                    is_recording = False
                    # Try to update menu even on error
                    try:
                        self.update_record_menu()
                        self.update_menu_states()
                    except:
                        pass
        except Exception as e:
            print(f"[CRITICAL] Error in toggle_recording: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            # Reset to safe state
            trigger_key_held = False
            is_recording = False
            status_queue.put(f"{APP_ICON_ERROR} Error")

    @rumps.clicked("Select Model", "Reload Current Model")
    def reload_model(self, _):
         """Callback to reload the currently selected model."""
         if selected_model_name:
              print(f"Reloading model: {selected_model_name}")
              self.change_model(None, model_name=selected_model_name)
         else:
              print("No model selected to reload.")
              self.title = f"{APP_ICON_ERROR} No Model"

    def auto_select_model(self, _):
        """Auto-select best model based on Mac capabilities."""
        global selected_model_name
        optimal_model = detect_optimal_model()
        selected_model_name = optimal_model
        self.change_model(None, model_name=optimal_model)
        rumps.notification(title="Model Auto-Selected", subtitle=f"Selected: {optimal_model}", message="")
    
    def select_cloud_provider(self, sender, provider_name):
        """Select cloud API provider."""
        try:
            from config import CLOUD_PROVIDER
            update_config_value('CLOUD_PROVIDER', f'"{provider_name.lower()}"')
            update_config_value('USE_CLOUD_API', 'True')
            rumps.notification(title="Cloud API Selected", subtitle=f"Using {provider_name}", message="")
            # Update menu states
            self.create_model_submenu()
        except Exception as e:
            print(f"[Error] Failed to select cloud provider: {e}")
    
    def create_shortcuts_submenu(self):
        """Create keyboard shortcuts submenu."""
        try:
            from config import (
                TRIGGER_KEY, TRIGGER_KEY_2, TOGGLE_RECORDING_KEY, QUIT_APP_KEY,
                RELOAD_MODEL_KEY, TOGGLE_APP_KEY
            )
            
            # Format trigger key display
            trigger_display = TRIGGER_KEY
            if TRIGGER_KEY_2 and TRIGGER_KEY_2.lower() != 'none':
                trigger_display = f"{TRIGGER_KEY} or {TRIGGER_KEY_2}"
            
            shortcuts_info = [
                ("Push-to-Talk", trigger_display),
                ("Toggle Recording", TOGGLE_RECORDING_KEY),
                ("Reload Model", RELOAD_MODEL_KEY),
                ("Toggle App", TOGGLE_APP_KEY),
                ("Quit App", QUIT_APP_KEY),
            ]
            
            for name, key_str in shortcuts_info:
                item = rumps.MenuItem(f"{name}: {key_str}", callback=None)
                item.set_callback(None)  # Non-clickable
                self.shortcuts_menu[name] = item
        except Exception as e:
            print(f"[Warning] Failed to create shortcuts submenu: {e}")

    def change_model(self, sender, model_name=None):
        """Callback function when a model is selected from the menu."""
        global selected_model_name, transcription_pipe, faster_whisper_model, current_model_id

        if model_name is None and sender is not None:
            model_name = sender.title

        if not model_name or model_name not in MODEL_MAP:
            print(f"Invalid model name: {model_name}", file=sys.stderr)
            return

        # Check if model is already loaded (either Faster Whisper or regular)
        model_already_loaded = False
        if USE_FASTER_WHISPER and FASTER_WHISPER_AVAILABLE:
            model_already_loaded = (model_name == selected_model_name and faster_whisper_model is not None)
        else:
            model_already_loaded = (model_name == selected_model_name and transcription_pipe is not None)
        
        if model_already_loaded:
            print(f"Model '{model_name}' is already loaded.")
            # Ensure state is correct in menu
            for name, item in self.model_menu.items():
                item.state = (name == model_name)
            return

        print(f"Changing model to: {model_name}")
        selected_model_name = model_name
        target_model_id = MODEL_MAP[model_name]

        # Update menu state
        for name, item in self.model_menu.items():
            item.state = (name == model_name)

        # Clear current models and trigger load in background thread
        transcription_pipe = None
        faster_whisper_model = None
        current_model_id = None
        if self.load_thread and self.load_thread.is_alive():
             print("Waiting for previous model load to complete...")
             # Optionally, implement a way to cancel the previous thread if needed
             # For now, we just let it finish to avoid complexity.
             self.load_thread.join(timeout=5.0) # Wait briefly

        self.load_thread = threading.Thread(target=load_model_and_processor, args=(target_model_id,), daemon=True)
        self.load_thread.start()


    def check_queues(self, _):
        """Timer callback to check status and transcription queues."""
        try:
            # Update status bar title
            status_update = status_queue.get_nowait()
            self.title = status_update
            status_queue.task_done()
        except queue.Empty:
            pass # No status update

        try:
            # Handle completed transcription
            transcription = transcription_queue.get_nowait()
            print(f"[Info] Received transcription: \'{transcription[:50]}...\'")

            # --- Copy to Clipboard (ALWAYS) and Try to Paste ---
            try:
                print(f"[Info] Transcription received: '{transcription}'")
                print("[Info] Copying transcription to clipboard...")
                
                # ALWAYS copy to clipboard first (this works even if paste fails)
                pyperclip.copy(transcription)

                # Verify copy worked
                clipboard_check = pyperclip.paste()
                if clipboard_check == transcription or clipboard_check[:50] == transcription[:50]:
                    print("[Info] ✅ Successfully copied to clipboard!")
                else:
                    print("[Warning] Clipboard copy may have failed, retrying...")
                    pyperclip.copy(transcription)
                
                # Show notification with transcribed text (so user knows what was transcribed)
                text_preview = transcription[:80] + "..." if len(transcription) > 80 else transcription
                # Don't show notification here - show it after paste attempt so user knows if paste worked
                
                # Direct paste at cursor position - optimized for production
                try:
                    from config import PASTE_DELAY, ENABLE_DIRECT_PASTE, PASTE_RETRY_ATTEMPTS
                    paste_delay = PASTE_DELAY
                    enable_direct = ENABLE_DIRECT_PASTE
                    retry_attempts = PASTE_RETRY_ATTEMPTS
                except:
                    paste_delay = 0.1
                    enable_direct = True
                    retry_attempts = 3
                
                # Minimal delay for direct paste - ensure we paste immediately
                time.sleep(paste_delay)
                paste_success = False

                # Always try to paste directly (not just clipboard)
                try:
                    from config import FORCE_DIRECT_PASTE
                    force_paste = FORCE_DIRECT_PASTE
                except:
                    force_paste = True  # Default to forcing direct paste

                if enable_direct or force_paste:
                    # Method 1: Enhanced AppleScript paste - ensures focus and pastes at cursor
                    # This method focuses the frontmost app and pastes at the exact cursor position
                    try:
                        log_info("[Info] Attempting enhanced paste at cursor position...")
                        import subprocess
                        
                        # Enhanced script that ensures focus and pastes at cursor
                        # This will paste wherever the text cursor (caret) is positioned
                        enhanced_paste_script = '''
                            tell application "System Events"
                                -- Get the frontmost application
                                set frontApp to name of first application process whose frontmost is true
                                
                                -- Activate the application to ensure it has focus
                                tell process frontApp
                                    set frontmost to true
                                    delay 0.1
                                    
                                    -- Paste at cursor position (Cmd+V)
                                    -- This will paste wherever the text cursor is in the active input field
                                    keystroke "v" using command down
                                end tell
                            end tell
                        '''
                        
                        for attempt in range(retry_attempts):
                            # Small delay before each attempt to ensure system is ready
                            if attempt > 0:
                                time.sleep(0.15)
                            
                            result = subprocess.run(['osascript', '-e', enhanced_paste_script], 
                                                   check=False, timeout=3, 
                                                   capture_output=True)
                            if result.returncode == 0:
                                log_info(f"[Info] ✅ Paste sent to cursor position (attempt {attempt+1})")
                                paste_success = True
                                time.sleep(0.15)  # Brief pause to let paste complete
                                break
                            else:
                                log_info(f"[Info] Paste attempt {attempt+1} returned code {result.returncode}")
                                if result.stderr:
                                    log_info(f"[Info] Error: {result.stderr.decode('utf-8', errors='ignore')}")
                    
                    except Exception as e:
                        log_info(f"[Info] Enhanced AppleScript paste error: {e}")
                    
                    # Method 2: Direct pynput paste with focus management (fallback)
                    if not paste_success:
                        try:
                            log_info("[Info] Attempting direct pynput paste at cursor...")
                            
                            # Ensure we're pasting to the active window/focused input field
                            # The cursor position is automatically where we paste
                            time.sleep(0.15)
                            
                            # Try multiple times with proper timing for maximum reliability
                            for attempt in range(retry_attempts):
                                try:
                                    # Direct Cmd+V paste - works in any text field, input, or space
                                    # This will paste at the text cursor (caret) position in the active input
                                    # The paste happens wherever the cursor is positioned
                                    keyboard_controller.press(keyboard.Key.cmd)
                                    time.sleep(0.08)  # Slightly longer delay for reliability
                                    keyboard_controller.press('v')
                                    time.sleep(0.08)
                                    keyboard_controller.release('v')
                                    keyboard_controller.release(keyboard.Key.cmd)
                                    
                                    log_info(f"[Info] ✅ Direct paste sent to cursor (attempt {attempt+1})")
                                    paste_success = True
                                    
                                    # Brief pause to let paste complete
                                    time.sleep(0.15)
                                    break
                                except Exception as e:
                                    log_info(f"[Info] Paste attempt {attempt+1} failed: {e}")
                                    if attempt < retry_attempts - 1:
                                        time.sleep(0.15)  # Wait before retry
                                    continue
                        except Exception as e:
                            log_info(f"[Info] pynput paste error: {e}")

                    # Method 3: Universal AppleScript paste (final fallback)
                    if not paste_success:
                        try:
                            log_info("[Info] Attempting universal AppleScript paste...")
                            import subprocess
                            # Universal paste method - pastes at cursor in any app
                            universal_script = '''
                                tell application "System Events"
                                    -- Ensure we're pasting to the active window
                                    set frontmost of first process whose frontmost is true to true
                                    delay 0.1
                                    -- Paste at cursor position
                                    keystroke "v" using command down
                                end tell
                            '''
                            result = subprocess.run(['osascript', '-e', universal_script], 
                                                   check=False, timeout=3, 
                                                   capture_output=True)
                            if result.returncode == 0:
                                log_info("[Info] ✅ Universal paste method succeeded")
                                paste_success = True
                                time.sleep(0.15)  # Brief pause to let paste complete
                        except Exception as e:
                            log_info(f"[Info] Universal paste method failed: {e}")
                else:
                    # Legacy paste method (if direct paste is disabled)
                    log_info("[Info] Direct paste disabled, using clipboard only")

                if paste_success:
                    log_info("[Info] ✅ Text pasted directly at cursor position")
                    # Minimal notification (production mode)
                    try:
                        from config import SHOW_NOTIFICATIONS
                        if SHOW_NOTIFICATIONS:
                            rumps.notification(
                                title="✅ Pasted", 
                                subtitle="Text inserted", 
                                message=f"{text_preview}"
                            )
                    except:
                        pass
                else:
                    # If direct paste failed, text is still in clipboard as backup
                    log_info("[Info] ⚠️  Direct paste failed - text is in clipboard (press Cmd+V if needed)")
                    # Only show notification if paste completely failed
                    try:
                        from config import SHOW_NOTIFICATIONS
                        if SHOW_NOTIFICATIONS:
                            rumps.notification(
                                title="📋 In Clipboard", 
                                subtitle="Press Cmd+V to paste", 
                                message=f"{text_preview}"
                            )
                    except:
                        pass

            except Exception as paste_error:
                print(f"[Error] Failed to copy or simulate paste: {paste_error}", file=sys.stderr)
                # Keep the clipboard copy attempt even if paste fails
                if pyperclip.paste() != transcription: # Check if copy failed before paste attempt
                    try:
                         pyperclip.copy(transcription)
                         print("[Info] Retrying copy to clipboard after paste error.")
                         rumps.notification(title="Transcription Copied", subtitle="Paste failed", message="Copied to clipboard instead.")
                    except Exception as copy_error:
                         print(f"[Error] Could not copy transcription to clipboard after paste failure: {copy_error}", file=sys.stderr)
                         self.title = f"{APP_ICON_ERROR} Copy Failed"
                else:
                     # Copy succeeded, but paste failed
                     rumps.notification(title="Transcription Copied", subtitle="Paste failed", message="Copied to clipboard instead.")
                     self.title = f"{APP_ICON_ERROR} Paste Failed"


            transcription_queue.task_done()

            # Reset status to idle after successful handling (paste or copy)
            if not self.title.startswith(APP_ICON_TRANSCRIBING) and not self.title.startswith(APP_ICON_ERROR):
                 self.title = f"{APP_ICON_IDLE} Ready ({selected_model_name})"

        except queue.Empty:
            pass # No transcription result

    def configure_dock_visibility(self):
        """Configure dock visibility after app initialization."""
        try:
            import AppKit
            if AppKit.NSApp is not None:
                AppKit.NSApp.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
                print("[Info] App configured as status bar only (hidden from dock)")
            else:
                print("[Warning] NSApp still not available - app will appear in dock")
        except ImportError:
            print("[Warning] AppKit not available - running with dock icon")
        except Exception as e:
            print(f"[Warning] Could not configure dock visibility: {e}")

    def quit_application(self, _):
        """Cleanly stop listener and quit."""
        global keyboard_listener
        print("Quit button clicked.")
        if keyboard_listener:
            print("Stopping keyboard listener...")
            keyboard_listener.stop()
        if audio_stream: # Ensure audio stream is closed
            try:
                audio_stream.stop()
                audio_stream.close()
            except Exception as e:
                print(f"Error closing audio stream on quit: {e}", file=sys.stderr)
        print("Quitting application.")
        rumps.quit_application()

# --- Model Preloading Functions ---
def preload_essential_models():
    """Preload Tiny model in background for instant switching."""
    global preload_thread
    if preload_thread and preload_thread.is_alive():
        return
    
    preload_thread = threading.Thread(target=_preload_models_background, daemon=True)
    preload_thread.start()

def _preload_models_background():
    """Background thread to preload models."""
    try:
        print("[Preload] Starting background model preloading...")
        
        # Preload Tiny model if it exists
        priority_models = []
        if "Tiny (.en)" in MODEL_MAP:
            priority_models.append(("Tiny (.en)", MODEL_MAP["Tiny (.en)"]))
        
        if not priority_models:
            print("[Preload] No priority models found in MODEL_MAP, skipping preload")
            return
        
        for model_name, model_id in priority_models:
            if model_id not in cached_models and model_id not in cached_faster_whisper_models:
                try:
                    print(f"[Preload] Loading {model_name} model...")
                    result = load_model_and_processor(model_id, cache_only=True)
                    if result:
                        cached_models[model_id] = result
                        print(f"[Preload] ✅ {model_name} model cached successfully")
                    else:
                        print(f"[Preload] ❌ Failed to cache {model_name} model")
                except Exception as e:
                    print(f"[Preload] Error caching {model_name}: {e}")
        
        print(f"[Preload] Background preloading complete. {len(cached_models)} models cached.")
        
    except Exception as e:
        print(f"[Preload] Background preloading failed: {e}")

def get_cached_model(model_id):
    """Get model from cache if available."""
    return cached_models.get(model_id)

def unload_models():
    """Unload models to free memory (for battery/memory optimization)."""
    global transcription_pipe, faster_whisper_model, current_model_id, vad_model
    
    try:
        from config import AUTO_UNLOAD_MODELS
        if not AUTO_UNLOAD_MODELS:
            return
    except:
        pass
    
    print("[Memory] Unloading models to free memory...")
    
    # Unload transcription pipeline
    if transcription_pipe is not None:
        transcription_pipe = None
    
    # Unload Faster Whisper model
    if faster_whisper_model is not None:
        faster_whisper_model = None
    
    # Unload VAD model
    if vad_model is not None:
        vad_model = None
    
    current_model_id = None
    
    # Clear caches if needed (optional - comment out to keep cached)
    # cached_models.clear()
    # cached_faster_whisper_models.clear()
    
    print("[Memory] Models unloaded")

def enable_streaming_mode():
    """Enable streaming processing for real-time transcription."""
    global streaming_enabled
    try:
        from config import PROCESSING_MODE
        # Enable streaming for Optimized and Ultra-Fast modes
        streaming_enabled = PROCESSING_MODE in ["Ultra-Fast", "Optimized"]
        print(f"[Streaming] Enabled: {streaming_enabled} for mode: {PROCESSING_MODE}")
    except:
        streaming_enabled = False

def process_streaming_chunk(audio_chunk):
    """Process a chunk of audio for real-time streaming."""
    try:
        if transcription_pipe is None or len(audio_chunk) < 1600:  # Too short
            return ""
        
        # Quick processing for streaming
        result = transcription_pipe({
            "raw": audio_chunk.astype(np.float32),
            "sampling_rate": get_optimal_sample_rate()
        })
        
        text = result.get('text', '').strip() if result else ''
        return text
        
    except Exception as e:
        print(f"[Streaming] Error processing chunk: {e}")
        return ""

# --- Main Execution ---
if __name__ == "__main__":
    # CRASH PREVENTION: Wrap entire app in crash recovery
    max_restart_attempts = 3
    restart_count = 0
    
    while restart_count < max_restart_attempts:
        try:
            # Check for accessibility permissions early (macOS specific)
            if sys.platform == "darwin":
                if not os.environ.get('EVENTTAP_TRUSTED'):
                    print("--- Accessibility Permissions Needed ---", file=sys.stderr)
                    print("This app needs Accessibility permissions to listen for global key presses (like Option key).", file=sys.stderr)
                    print("Please go to System Settings > Privacy & Security > Accessibility.", file=sys.stderr)
                    print("Click the '+' button, find your terminal application (e.g., Terminal, iTerm) or the application you used to launch this script (e.g., Python launcher, VS Code), and add it to the list.", file=sys.stderr)
                    print("You might need to restart this script after granting permissions.", file=sys.stderr)
                    # Simplified print statement
                    print("----------------------------------------", file=sys.stderr)
                    print("\n", file=sys.stderr) # Add newline separately
                    # Note: We can't programmatically check if we have permissions,
                    # pynput handles the error if not trusted, but this warning helps the user.

            # Load VAD model at startup (only if not disabled)
            try:
                from config import DISABLE_BACKGROUND_PRELOADING
                if not DISABLE_BACKGROUND_PRELOADING:
                    load_vad_model()
                    # Start model preloading in background for instant switching
                    print("[Info] Starting background model preloading for instant mode switching...")
                    preload_essential_models()
            except:
                load_vad_model()
                preload_essential_models()

            app = WhisperStatusBarApp()
            app.run()
            # If app.run() returns normally, exit the restart loop
            break
            
        except KeyboardInterrupt:
            # User requested exit - don't restart
            print("\n[Info] Shutting down gracefully...")
            break
        except Exception as e:
            # CRASH PREVENTION: Catch any crash and attempt recovery
            restart_count += 1
            print(f"\n[CRITICAL] App crashed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            
            if restart_count < max_restart_attempts:
                print(f"[Recovery] Attempting restart {restart_count}/{max_restart_attempts}...")
                # Clean up any global state (access module-level globals directly)
                try:
                    # Reset recording state
                    import main
                    if hasattr(main, 'is_recording'):
                        main.is_recording = False
                    # Clean up audio stream
                    if hasattr(main, 'audio_stream') and main.audio_stream is not None:
                        try:
                            if hasattr(main.audio_stream, 'active') and main.audio_stream.active:
                                main.audio_stream.stop()
                        except:
                            pass
                        try:
                            main.audio_stream.close()
                        except:
                            pass
                        main.audio_stream = None
                except Exception as cleanup_error:
                    print(f"[Warning] Cleanup error: {cleanup_error}")
                # Wait a moment before restarting
                import time
                time.sleep(2)
            else:
                print(f"[FATAL] App crashed {max_restart_attempts} times. Exiting.", file=sys.stderr)
                break 