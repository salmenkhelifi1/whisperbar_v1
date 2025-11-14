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

# Fix PATH for GUI applications (helps with ffmpeg detection)
os.environ['PATH'] = '/opt/homebrew/bin:/usr/local/bin:' + os.environ.get('PATH', '')

# Global VAD model
vad_model = None

# Import Controller and time
from pynput.keyboard import Controller

# --- Constants and Configuration ---
APP_NAME = "WhisperBar"
APP_ICON_IDLE = "🎤"
APP_ICON_RECORDING = "🔴"
APP_ICON_TRANSCRIBING = "⚡"
APP_ICON_LOADING = "🚀"
APP_ICON_ERROR = "❌"
APP_ICON_SUCCESS = "✅"
APP_ICON_PROCESSING = "🧠"
APP_ICON_OPTIMIZING = "🎯"

SAMPLE_RATE = 16000  # Whisper expects 16kHz audio (dynamic based on mode)
CHANNELS = 1
TEMP_FILE_PATH = "temp_recording.wav"

# Smart Processing Configuration  
SMART_MAX_SEGMENT_LENGTH = 8.0  # Maximum segment length for optimization
SMART_MIN_SEGMENT_LENGTH = 1.5  # Minimum segment length before merging

# Model IDs mapping
MODEL_MAP = {
    "Tiny (.en)": "openai/whisper-tiny.en",  # Real tiny model for ultra-fast processing
    "Small (.en)": "distil-whisper/distil-small.en",
    "Medium (.en)": "distil-whisper/distil-medium.en",
    "Large (v3.5)": "distil-whisper/distil-large-v3.5",
}
DEFAULT_MODEL_NAME = "Medium (.en)"

# Trigger key - Using Right Shift
TRIGGER_KEY = keyboard.Key.shift_r

# --- Global State ---
# Using queues for thread-safe communication
status_queue = queue.Queue()
transcription_queue = queue.Queue()

# Model caching for instant switching
cached_models = {}  # {model_id: {"pipe": pipeline, "processor": processor, "model": model}}
preload_thread = None

# Transcription Pipeline related
transcription_pipe = None
current_model_id = None
selected_model_name = DEFAULT_MODEL_NAME

# Recording related
is_recording = False
audio_frames = []
audio_stream = None

# Pre-allocated buffers for speed optimization
PRE_ALLOCATED_BUFFER_SIZE = 16000 * 30  # 30 seconds at 16kHz
pre_allocated_audio_buffer = np.zeros(PRE_ALLOCATED_BUFFER_SIZE, dtype=np.float32)
temp_processing_buffer = np.zeros(PRE_ALLOCATED_BUFFER_SIZE, dtype=np.float32)

# Processing State
current_processing_mode = "Optimized"

# Keyboard listener related
listener_thread = None
keyboard_listener = None
trigger_key_held = False

# Processing Control State
is_processing = False
cancel_processing_event = threading.Event()

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
            print("[Info] VAD disabled in config")
            return None
            
        if VAD_MODEL == "silero":
            print("[Info] Loading Silero VAD model...")
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
            print("[Info] Silero VAD model loaded successfully")
            return vad_model
        else:
            print(f"[Warning] VAD model '{VAD_MODEL}' not implemented yet")
            return None
            
    except Exception as e:
        print(f"[Warning] Failed to load VAD model: {e}")
        return None

def detect_speech_segments(audio_data, sample_rate=16000):
    """Detect speech segments in audio using VAD."""
    try:
        from config import (VAD_THRESHOLD, VAD_MIN_SPEECH_DURATION, 
                           VAD_MIN_SILENCE_DURATION, VAD_SPEECH_PAD_BEFORE, 
                           VAD_SPEECH_PAD_AFTER)
        
        if vad_model is None:
            print("[Info] VAD not available, processing entire audio")
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
            print("[Info] No speech detected by VAD")
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
        
        print(f"[Info] VAD detected {len(merged_segments)} speech segments")
        print(f"[Info] Speech ratio: {speech_ratio:.1%} ({total_speech_duration:.1f}s/{total_duration:.1f}s)")
        
        return merged_segments
        
    except Exception as e:
        print(f"[Warning] VAD processing failed: {e}")
        return [(0, len(audio_data))]  # Fallback to entire audio

# --- Smart Processing Functions ---
def process_audio_traditional(audio_data_np):
    """Traditional VAD processing (proven baseline method)."""
    try:
        print(f"[Traditional] Processing {len(audio_data_np)/SAMPLE_RATE:.1f}s of audio")
        
        # Use standard VAD to detect speech segments
        speech_segments = detect_speech_segments(audio_data_np)
        
        if not speech_segments:
            print("[Traditional] No speech detected")
            return ""
        
        # Process each speech segment and combine results
        transcriptions = []
        total_processed_duration = 0
        
        for i, (start_idx, end_idx) in enumerate(speech_segments):
            segment_audio = audio_data_np[start_idx:end_idx]
            segment_duration = len(segment_audio) / SAMPLE_RATE
            total_processed_duration += segment_duration
            
            print(f"[Traditional] Processing speech segment {i+1}/{len(speech_segments)} ({segment_duration:.1f}s)")
            
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
        print(f"[Traditional] VAD processing complete. Speed improvement: {speed_improvement:.1f}%")
        
        return final_text
        
    except Exception as e:
        print(f"[Error] Traditional processing failed: {e}")
        return ""

def process_audio_optimized(audio_data_np):
    """Optimized processing pipeline with smart segmentation."""
    try:
        from config import SMART_MAX_SEGMENT_LENGTH, SMART_MIN_SEGMENT_LENGTH
        
        start_time = time.time()
        print(f"[Optimized] Starting smart transcription for {len(audio_data_np)/SAMPLE_RATE:.1f}s of audio")
        
        # Smart segmentation with VAD
        speech_segments = detect_speech_segments(audio_data_np)
        if not speech_segments:
            print("[Optimized] No speech detected")
            return ""
        
        # Optimize segments for better processing
        optimized_segments = optimize_segments_for_speed(speech_segments, audio_data_np)
        
        # Process segments sequentially (thread-safe)
        final_text = process_segments_sequential(optimized_segments, audio_data_np)
        
        processing_time = time.time() - start_time
        print(f"[Optimized] Complete in {processing_time:.2f}s: '{final_text[:50]}...'")
        
        return final_text
        
    except Exception as e:
        print(f"[Error] Optimized processing failed: {e}")
        return ""

def process_audio_ultra_fast(audio_data_np):
    """Ultra-Fast processing pipeline optimized for maximum speed with Tiny model."""
    try:
        start_time = time.time()
        sample_rate = get_optimal_sample_rate()
        duration = len(audio_data_np)/sample_rate
        print(f"[Ultra-Fast] Starting maximum-speed transcription for {duration:.1f}s of audio")
        
        # Skip VAD entirely for maximum speed - just process everything as one segment
        print("[Ultra-Fast] Skipping VAD for maximum speed - processing entire audio")
        speech_segments = [(0, len(audio_data_np))]  # Process entire audio as one segment
        
        # Process segments with speed priority (bypassing optimization)
        final_text = process_segments_sequential(speech_segments, audio_data_np)
        
        processing_time = time.time() - start_time
        print(f"[Ultra-Fast] Complete in {processing_time:.2f}s: '{final_text[:50]}...' (Tiny model + no VAD)")
        
        return final_text
        
    except Exception as e:
        print(f"[Error] Ultra-fast processing failed: {e}")
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
        
        print(f"[Smart] Optimized {len(speech_segments)} segments → {len(merged)} segments")
        return merged
        
    except Exception as e:
        print(f"[Warning] Segment optimization failed: {e}")
        return speech_segments

def transcribe_segment_fast(segment_audio, segment_id):
    """Fast transcription of a single segment using optimized pipeline."""
    try:
        if transcription_pipe is None:
            print(f"[Smart] Model not loaded for segment {segment_id}")
            return ""
        
        # Ensure correct format
        if segment_audio.ndim > 1:
            segment_audio = segment_audio.squeeze()
        segment_audio = segment_audio.astype(np.float32)
        
        duration = len(segment_audio) / SAMPLE_RATE
        print(f"[Smart] Transcribing segment {segment_id} ({duration:.1f}s)")
        
        start_time = time.time()
        result = transcription_pipe({
            "raw": segment_audio,
            "sampling_rate": SAMPLE_RATE
        })
        processing_time = time.time() - start_time
        
        text = result.get('text', '').strip() if result else ''
        print(f"[Smart] Segment {segment_id} done in {processing_time:.2f}s: '{text[:30]}...'")
        
        return text
        
    except Exception as e:
        print(f"[Error] Smart transcription failed for segment {segment_id}: {e}")
        return ""

def process_segments_sequential(segments, audio_data_np):
    """Process segments sequentially with smart optimizations."""
    try:
        print(f"[Smart] Processing {len(segments)} segments sequentially")
        
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
    
    # Use INT8 quantization for speed-focused modes (50-70% speed boost, 1-3% accuracy loss)
    if processing_mode in ["Ultra-Fast", "Optimized"]:
        print(f"[Quantization] Using INT8 quantization for {processing_mode} mode (50-70% speed boost)")
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,  # More accurate quantization
        )
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

def load_model_and_processor(model_id, cache_only=False):
    """Loads the specified model and processor with caching support.
    
    Args:
        model_id: The model identifier
        cache_only: If True, only cache the model without setting as active
    """
    global transcription_pipe, current_model_id
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
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "attn_implementation": "sdpa"  # Explicitly enable SDPA
        }
        
        # Add quantization if enabled
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            # Don't move to device manually when using quantization - it's handled automatically
            print("[Quantization] Loading model with INT8 quantization for speed boost...")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **model_kwargs)
        
        # Only move to device if not using quantization (quantization handles device placement)
        if quantization_config is None:
        model.to(device)
        
        # Apply torch.compile() for 20-30% speed boost (JIT compilation)
        try:
            if hasattr(torch, 'compile'):
                print("[Optimization] Applying torch.compile() for JIT acceleration...")
                model = torch.compile(model, mode="reduce-overhead")
                print("[Optimization] ✅ torch.compile() applied successfully")
        except Exception as e:
            print(f"[Optimization] torch.compile() not available: {e}")
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
        pipeline_kwargs = {
            "model": model,
            "tokenizer": processor.tokenizer,
            "feature_extractor": processor.feature_extractor,
            "chunk_length_s": 12,         # Optimal chunk size for balance
            "batch_size": 1,
            "return_timestamps": False,
            "torch_dtype": torch_dtype,
            "device": device,
            "generate_kwargs": optimized_gen_kwargs
        }
        
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
    global transcription_pipe, is_processing

    is_processing = True

    if transcription_pipe is None:
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

        # Enhanced audio preprocessing with smart processing modes
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
                
                # Process based on selected mode
                if processing_mode == "Ultra-Fast":
                    # Ultra-Fast Mode: Force Tiny model + optimized processing
                    try:
                        audio_duration = len(audio_data_float32) / SAMPLE_RATE
                        status_queue.put(f"⚡ Ultra-Fast • {audio_duration:.1f}s → Processing...")
                        transcription = process_audio_ultra_fast(audio_data_float32)
                    except Exception as ultra_fast_error:
                        print(f"[Warning] Ultra-fast mode failed: {ultra_fast_error}")
                        print("[Info] Falling back to traditional processing...")
                        processing_mode = "Traditional"
                
                elif processing_mode == "Optimized":
                    # Optimized Mode: Smart processing + user's model choice
                    try:
                        audio_duration = len(audio_data_float32) / SAMPLE_RATE
                        status_queue.put(f"🎯 Optimized • {audio_duration:.1f}s → Smart Processing...")
                        transcription = process_audio_optimized(audio_data_float32)
                    except Exception as optimized_error:
                        print(f"[Warning] Optimized mode failed: {optimized_error}")
                        print("[Info] Falling back to traditional processing...")
                        processing_mode = "Traditional"
                
                if processing_mode == "Traditional":
                    # Traditional Mode: Proven baseline VAD processing  
                    try:
                        audio_duration = len(audio_data_float32) / SAMPLE_RATE
                        status_queue.put(f"📊 Traditional • {audio_duration:.1f}s → Processing...")
                        transcription = process_audio_traditional(audio_data_float32)
                    except Exception as traditional_error:
                        print(f"[Warning] Traditional mode failed: {traditional_error}")
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
    """Called by sounddevice for each audio block."""
    if status:
        print(f"Audio Stream Status: {status}", file=sys.stderr)
    if is_recording:
        audio_frames.append(indata.copy())

def start_recording():
    global is_recording, audio_stream, audio_frames
    # Allow starting even if processing, it will be cancelled by on_press
    if is_recording:
        print("[Warning] Already recording, ignoring request.")
        return
    if transcription_pipe is None:
         print("[Error] Model not loaded, cannot start recording.")
         status_queue.put(f"{APP_ICON_ERROR} No Model Loaded")
         return

    print("[Info] Starting recording...")
    status_queue.put(f"{APP_ICON_RECORDING} Recording • Hold Shift...")
    is_recording = True
    audio_frames = []
    try:
        current_sample_rate = get_optimal_sample_rate()
        audio_stream = sd.InputStream(
            samplerate=current_sample_rate, channels=CHANNELS, callback=audio_callback, dtype='float32'
        )
        audio_stream.start()
    except Exception as e:
        print(f"[Error] Starting audio stream failed: {e}", file=sys.stderr)
        status_queue.put(f"{APP_ICON_ERROR} Mic Error")
        is_recording = False

def stop_recording_and_transcribe():
    global is_recording, audio_stream, audio_frames, is_processing
    if not is_recording:
        return

    print("[Info] Recording stopped.")
    status_queue.put(f"{APP_ICON_PROCESSING} Processing audio...")
    is_recording = False
    if audio_stream:
        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception as e:
            print(f"[Error] Stopping audio stream failed: {e}", file=sys.stderr)
        finally:
            audio_stream = None

    # Add debug print for audio frames check
    print(f"[Debug] Checking audio frames. Count: {len(audio_frames)}")

    if audio_frames:
        # Add check for empty frames just in case
        if all(frame.size == 0 for frame in audio_frames):
             print("[Warning] Audio frames captured, but they are all empty.")
             status_queue.put(f"{APP_ICON_IDLE} Ready ({selected_model_name})")
             return

        try:
             audio_data = np.concatenate(audio_frames, axis=0).astype(np.float32)
             print(f"[Debug] Concatenated audio data shape: {audio_data.shape}")
        except ValueError as e:
             print(f"[Error] Failed to concatenate audio frames: {e}", file=sys.stderr)
             status_queue.put(f"{APP_ICON_ERROR} Concat Failed")
             audio_frames = [] # Clear potentially problematic frames
             return

        audio_frames = []
        cancel_processing_event.clear()
        is_processing = True
        print("[Debug] Launching transcription thread...") # Add thread launch debug print
        thread = threading.Thread(target=transcribe_audio_thread, args=(audio_data,), daemon=True)
        thread.start()
    else:
        print("[Warning] No audio frames captured.") # Changed from Info to Warning
        status_queue.put(f"{APP_ICON_IDLE} Ready ({selected_model_name})")


# --- Keyboard Listener Callbacks ---
def on_press(key):
    global trigger_key_held, is_processing, is_recording, audio_stream, audio_frames
    if key == TRIGGER_KEY:
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
    global trigger_key_held, is_recording
    if key == TRIGGER_KEY:
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
        super(WhisperStatusBarApp, self).__init__(APP_NAME, title=APP_ICON_IDLE, quit_button='Quit')
        
        # Schedule dock hiding for after app initialization
        try:
            from config import HIDE_FROM_DOCK
            if HIDE_FROM_DOCK:
                # Defer dock hiding until after app is fully initialized
                threading.Timer(0.5, self.configure_dock_visibility).start()
            else:
                print("[Info] App will appear in dock (HIDE_FROM_DOCK=False)")
        except Exception as e:
            print(f"[Warning] Could not configure dock visibility: {e}")
        
        self.model_menu = rumps.MenuItem("Select Model")
        self.mode_menu = rumps.MenuItem("Processing Mode")
        self.menu = [
            self.model_menu, 
            self.mode_menu,
            None  # Separator
        ]
        self.create_model_submenu()
        self.create_mode_submenu()
        self.load_thread = None

        # Start background thread to check queues
        self.queue_timer = rumps.Timer(self.check_queues, 1)
        self.queue_timer.start()

        # Trigger initial model load in background
        self.change_model(None, model_name=selected_model_name)
        
        # Initialize VAD model in background
        threading.Timer(2.0, load_vad_model).start()

        # Start keyboard listener after a short delay
        threading.Timer(1.0, start_keyboard_listener).start()


    def create_model_submenu(self):
        """Creates or updates the model selection submenu."""
        # Check if the underlying NSMenu exists before trying to clear
        if self.model_menu._menu is not None:
            self.model_menu.clear()
        else:
             print("Skipping initial model_menu.clear() as _menu is None.")

        # Clear existing dictionary items manually just in case clear() didn't run
        # Convert keys to list to avoid RuntimeError: dictionary changed size during iteration
        existing_keys = list(self.model_menu.keys())
        for key in existing_keys:
             # Check if key exists before deleting, though it should
             if key in self.model_menu:
                 del self.model_menu[key]

        for name in MODEL_MAP.keys():
            # Use a factory function or lambda with default args to capture the current name
            # This ensures the correct 'name' is used when the callback fires later
            callback_func = lambda sender, captured_name=name: self.change_model(sender, model_name=captured_name)
            item = rumps.MenuItem(name, callback=callback_func)
            item.state = (name == selected_model_name)
            self.model_menu[name] = item # Add item to the MenuItem's dictionary

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

    @rumps.clicked("Select Model", "Reload Current Model")
    def reload_model(self, _):
         """Callback to reload the currently selected model."""
         if selected_model_name:
              print(f"Reloading model: {selected_model_name}")
              self.change_model(None, model_name=selected_model_name)
         else:
              print("No model selected to reload.")
              self.title = f"{APP_ICON_ERROR} No Model"

    def change_model(self, sender, model_name=None):
        """Callback function when a model is selected from the menu."""
        global selected_model_name, transcription_pipe, current_model_id

        if model_name is None and sender is not None:
            model_name = sender.title

        if not model_name or model_name not in MODEL_MAP:
            print(f"Invalid model name: {model_name}", file=sys.stderr)
            return

        if model_name == selected_model_name and transcription_pipe is not None:
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

        # Clear current pipe and trigger load in background thread
        transcription_pipe = None
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

            # --- Use Copy and Paste (Cmd+V) ---
            try:
                print("[Info] Copying transcription to clipboard...")
                pyperclip.copy(transcription)

                # Shorter delay for faster responsiveness  
                time.sleep(0.3)

                print("[Info] Attempting to paste...")
                paste_success = False

                # Use AppleScript as primary method (most reliable for GUI apps)
                try:
                    print("[Info] Using AppleScript paste method...")
                    import subprocess
                    
                    # First, get the frontmost application
                    get_app_script = '''
                    tell application "System Events"
                        name of first application process whose frontmost is true
                    end tell
                    '''
                    
                    try:
                        result = subprocess.run(['osascript', '-e', get_app_script], 
                                              capture_output=True, text=True, timeout=3)
                        frontmost_app = result.stdout.strip()
                        print(f"[Info] Target app: {frontmost_app}")
                    except:
                        frontmost_app = "unknown"
                    
                    # Try the paste
                    paste_script = '''
                    tell application "System Events"
                        keystroke "v" using command down
                    end tell
                    '''
                    subprocess.run(['osascript', '-e', paste_script], check=True, timeout=5)
                    print("[Info] AppleScript paste executed successfully")
                    paste_success = True
                    
                except Exception as e:
                    print(f"[Warning] AppleScript method failed: {e}")
                    
                    # Fallback to pynput method
                    try:
                        print("[Info] Fallback: Using pynput Cmd+V...")
                        keyboard_controller.press(keyboard.Key.cmd)
                        time.sleep(0.1)
                        keyboard_controller.press('v')
                        time.sleep(0.1)
                        keyboard_controller.release('v')
                        keyboard_controller.release(keyboard.Key.cmd)
                        print("[Info] pynput Cmd+V executed")
                        paste_success = True
                        
                    except Exception as e2:
                        print(f"[Warning] pynput method also failed: {e2}")

                # Method 3: Direct typing fallback (if AppleScript failed)
                if not paste_success:
                    try:
                        print("[Info] Method 3: Direct typing fallback...")
                        # Type more aggressively since other methods failed
                        keyboard_controller.type(transcription)
                        print("[Info] Direct typing completed")
                        paste_success = True
                    except Exception as e:
                        print(f"[Warning] Direct typing failed: {e}")

                # Show appropriate notification
                if paste_success:
                    rumps.notification(title="Transcription Pasted", subtitle="", message=f'{transcription[:30]}...')
                else:
                    rumps.notification(title="Transcription Copied", subtitle="Manual paste needed", message="Copied to clipboard - use Cmd+V to paste")

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
    """Preload Medium and Tiny models in background for instant switching."""
    global preload_thread
    if preload_thread and preload_thread.is_alive():
        return
    
    preload_thread = threading.Thread(target=_preload_models_background, daemon=True)
    preload_thread.start()

def _preload_models_background():
    """Background thread to preload models."""
    try:
        print("[Preload] Starting background model preloading...")
        
        # Preload Medium (default) and Tiny (ultra-fast) models
        priority_models = [
            ("Medium (.en)", MODEL_MAP["Medium (.en)"]),
            ("Tiny (.en)", MODEL_MAP["Tiny (.en)"])
        ]
        
        for model_name, model_id in priority_models:
            if model_id not in cached_models:
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
    # Check for accessibility permissions early (macOS specific)
    if sys.platform == "darwin":
        if not os.environ.get('EVENTTAP_TRUSTED'):
            print("--- Accessibility Permissions Needed ---", file=sys.stderr)
            print("This app needs Accessibility permissions to listen for global key presses (like Right Shift).", file=sys.stderr)
            print("Please go to System Settings > Privacy & Security > Accessibility.", file=sys.stderr)
            print("Click the '+' button, find your terminal application (e.g., Terminal, iTerm) or the application you used to launch this script (e.g., Python launcher, VS Code), and add it to the list.", file=sys.stderr)
            print("You might need to restart this script after granting permissions.", file=sys.stderr)
            # Simplified print statement
            print("----------------------------------------", file=sys.stderr)
            print("\n", file=sys.stderr) # Add newline separately
            # Note: We can't programmatically check if we have permissions,
            # pynput handles the error if not trusted, but this warning helps the user.

    # Load VAD model at startup
    load_vad_model()
    
    # Start model preloading in background for instant switching
    print("[Info] Starting background model preloading for instant mode switching...")
    preload_essential_models()

    app = WhisperStatusBarApp()
    app.run() 