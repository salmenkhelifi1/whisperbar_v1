"""
Utility functions for WhisperBar
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_macos_version():
    """Check if running on a supported macOS version"""
    if platform.system() != "Darwin":
        return False, "This application only works on macOS"
    
    version = platform.mac_ver()[0]
    major, minor = map(int, version.split('.')[:2])
    
    if major < 10 or (major == 10 and minor < 15):
        return False, f"macOS 10.15+ required. Found: {version}"
    
    return True, f"macOS {version}"

def check_accessibility_permissions():
    """
    Check if accessibility permissions are granted.
    Note: This is a basic check and may not be 100% accurate.
    """
    try:
        # Try to create a simple event tap to test permissions
        from Quartz import CGEventTapCreate, kCGHIDEventTap, kCGEventMaskForAllEvents
        
        def dummy_callback(proxy, type, event, refcon):
            return event
            
        event_tap = CGEventTapCreate(
            kCGHIDEventTap, 0, 0, kCGEventMaskForAllEvents, 
            dummy_callback, None
        )
        
        if event_tap:
            return True
        else:
            return False
    except:
        return False

def open_accessibility_settings():
    """Open System Settings to Accessibility section"""
    try:
        subprocess.run([
            "open", 
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        ])
        return True
    except:
        return False

def check_microphone_permissions():
    """Check if microphone permissions are granted"""
    try:
        import sounddevice as sd
        # Try to get default input device
        device = sd.default.device[0]
        return device is not None
    except:
        return False

def get_cache_size():
    """Get the size of the model cache directory"""
    from config import CACHE_DIR
    total_size = 0
    
    if os.path.exists(CACHE_DIR):
        for dirpath, dirnames, filenames in os.walk(CACHE_DIR):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
    
    # Convert to human readable format
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024.0:
            return f"{total_size:.1f} {unit}"
        total_size /= 1024.0
    return f"{total_size:.1f} TB"

def clear_cache():
    """Clear the model cache directory"""
    from config import CACHE_DIR
    import shutil
    
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
        return True
    return False

def get_model_info(model_name):
    """Get information about a model"""
    from config import MODEL_MAP
    
    model_id = MODEL_MAP.get(model_name)
    if not model_id:
        return None
        
    # Estimated sizes (these are approximate)
    size_map = {
        "distil-whisper/distil-tiny.en": "37 MB",
        "distil-whisper/distil-small.en": "244 MB", 
        "distil-whisper/distil-medium.en": "769 MB",
        "distil-whisper/distil-large-v3.5": "1.5 GB",
    }
    
    return {
        "name": model_name,
        "id": model_id,
        "size": size_map.get(model_id, "Unknown"),
        "type": "English-only" if model_id.endswith(".en") else "Multilingual"
    }

def format_duration(seconds):
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'torch', 'transformers', 'sounddevice', 'numpy', 
        'scipy', 'pynput', 'pyperclip', 'rumps'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, missing

def create_desktop_shortcut():
    """Create a desktop shortcut (macOS specific)"""
    # This would create a .app bundle or shell script
    # Implementation would depend on the specific requirements
    pass 