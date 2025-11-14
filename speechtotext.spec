# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# --- Main Script ---
main_script = 'main.py'

# --- Files to Bundle ---
datas = [
    ('config.py', '.'),
    ('utils.py', '.'), 
    ('README.md', '.'),
]

# --- Libraries/Modules ---
hiddenimports = [
    'objc', 
    'CoreFoundation', 
    'Quartz', 
    'Cocoa', 
    'Foundation', 
    'AppKit',
    'sounddevice',
    'transformers',
    'torch',
    'pyperclip',
    'rumps',
    'pynput',
    'numpy',
    'scipy',
    'accelerate',
    'datasets',
    'packaging',
    'packaging.version',
    'packaging.specifiers', 
    'packaging.requirements',
    'PIL',
    'scipy.special._cdflib',
    'scipy.integrate',
    'scipy.linalg',
]

# --- Analysis Settings ---
a = Analysis(
    [main_script],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib', 
        'IPython',
        'jupyter',
        'notebook',
        'setuptools',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    optimize=1,
)

# Filter out some large unnecessary binaries
a.binaries = [x for x in a.binaries if not any(name in x[0].lower() for name in [
    'libmkl', 'libblas', 'liblapack', 'tcl', 'tk'
])]

# --- Executable --- 
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WhisperBar',
    debug=False,
    bootloader_ignore_signals=False, 
    strip=False,
    upx=False,  # UPX can cause issues with PyTorch
    console=False,  # No console window for status bar app
    disable_windowed_traceback=False,
    argv_emulation=True,  # Important for macOS
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe, 
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='WhisperBar'
)

# --- macOS App Bundle ---
app = BUNDLE(
    coll,
    name='WhisperBar.app',
    bundle_identifier='com.whisperbar.app',
    info_plist={
        'CFBundleName': 'WhisperBar',
        'CFBundleDisplayName': 'WhisperBar', 
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSMicrophoneUsageDescription': 'WhisperBar needs microphone access to record audio for speech-to-text transcription.',
        'NSAppleEventsUsageDescription': 'WhisperBar needs to send keystrokes to paste transcribed text.',
        'LSUIElement': True,  # Hide from dock - status bar only
        'LSBackgroundOnly': False,
        'NSHighResolutionCapable': True,
        'CFBundleIconFile': 'app_icon',  # If you add an icon file
    }
) 