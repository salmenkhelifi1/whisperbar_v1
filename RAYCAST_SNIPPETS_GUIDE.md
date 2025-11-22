# Raycast Snippets Setup Guide

## 📁 Three Files Created

1. **`raycast_start.sh`** - Start WhisperBar
2. **`raycast_stop.sh`** - Stop WhisperBar  
3. **`raycast_restart.sh`** - Restart WhisperBar

## 🚀 How to Add to Raycast

### Step 1: Open Raycast Script Commands

1. Open Raycast (`Cmd+Space`)
2. Type `Extensions` and press Enter
3. Go to **Script Commands**
4. Click **+** to create new script

### Step 2: Add Start Script

**For `raycast_start.sh`:**

1. Click **Create Script**
2. Choose **Shell Script**
3. Fill in:
   - **Title:** `W`
   - **Subtitle:** `Start WhisperBar speech-to-text`
   - **Icon:** 🎤 (or choose custom)
   - **Keywords:** `whisper start`, `wb start`, `speech start`
   - **Script Path:** `/Users/salmenkhelifi/Documents/whisperbar_v1/raycast_start.sh`
   - **Mode:** `Run in Background` ✅
   - **Argument:** None

### Step 3: Add Stop Script

**For `raycast_stop.sh`:**

1. Click **Create Script** again
2. Choose **Shell Script**
3. Fill in:
   - **Title:** `WhisperBar Stop`
   - **Subtitle:** `Stop WhisperBar speech-to-text`
   - **Icon:** 🛑 (or choose custom)
   - **Keywords:** `whisper stop`, `wb stop`, `speech stop`
   - **Script Path:** `/Users/salmenkhelifi/Documents/whisperbar_v1/raycast_stop.sh`
   - **Mode:** `Run in Background` ✅
   - **Argument:** None

### Step 4: Add Restart Script

**For `raycast_restart.sh`:**

1. Click **Create Script** again
2. Choose **Shell Script**
3. Fill in:
   - **Title:** `WhisperBar Restart`
   - **Subtitle:** `Restart WhisperBar speech-to-text`
   - **Icon:** 🔄 (or choose custom)
   - **Keywords:** `whisper restart`, `wb restart`, `speech restart`
   - **Script Path:** `/Users/salmenkhelifi/Documents/whisperbar_v1/raycast_restart.sh`
   - **Mode:** `Run in Background` ✅
   - **Argument:** None

## 💡 Usage

Once set up:

1. Open Raycast (`Cmd+Space`)
2. Type your keyword (e.g., `whisper start`)
3. Press Enter
4. You'll see a notification confirming the action

## 📋 Quick Reference

| Action | Keyword Suggestions | File |
|--------|-------------------|------|
| **Start** | `whisper start`, `wb start` | `raycast_start.sh` |
| **Stop** | `whisper stop`, `wb stop` | `raycast_stop.sh` |
| **Restart** | `whisper restart`, `wb restart` | `raycast_restart.sh` |

## ✅ Features

- ✅ Shows macOS notifications when actions complete
- ✅ Checks if already running before starting
- ✅ Handles errors gracefully
- ✅ Works in background (doesn't block Raycast)
- ✅ Auto-detects Python (venv or system)

## 🔧 Troubleshooting

**Script not found?**

- Make sure the file paths are correct
- Check file permissions: `chmod +x raycast_*.sh`

**Notifications not showing?**

- Check macOS notification permissions for Raycast
- System Settings → Notifications → Raycast

**Script not executing?**

- Make sure Raycast has Accessibility permissions
- System Settings → Privacy & Security → Accessibility
