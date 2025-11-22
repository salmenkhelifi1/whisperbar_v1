# WhisperBar Raycast Script Commands

Three Script Commands for controlling WhisperBar from Raycast.

## 📁 Files

- `whisperbar-start.sh` + `whisperbar-start.json` - Start WhisperBar
- `whisperbar-stop.sh` + `whisperbar-stop.json` - Stop WhisperBar
- `whisperbar-restart.sh` + `whisperbar-restart.json` - Restart WhisperBar

## 🚀 Installation

### Method 1: Add Script Directory (Recommended)

1. Open Raycast (`Cmd+Space`)
2. Type `Extensions` and press Enter
3. Click the **+** button
4. Click **Add Script Directory**
5. Select: `/Users/salmenkhelifi/Documents/whisperbar_v1/raycast-scripts`
6. All three commands will appear in Raycast!

### Method 2: Create Individual Scripts

1. Open Raycast → Extensions → Script Commands
2. Click **+** → **Create Script Command**
3. For each script:
   - Choose **Shell Script**
   - **Script Path:** Point to the `.sh` file
   - Raycast will automatically detect the `.json` metadata file

## 💡 Usage

Once installed:

1. Open Raycast (`Cmd+Space`)
2. Type `WhisperBar` or `whisper`
3. Choose the action you want:
   - **WhisperBar Start** 🎤 - Start the application
   - **WhisperBar Stop** 🛑 - Stop the application
   - **WhisperBar Restart** 🔄 - Restart the application
4. Press Enter

## ✨ Features

- ✅ Proper Raycast metadata (JSON files)
- ✅ Icons for each command
- ✅ No-view mode (runs in background)
- ✅ Auto-detects Python (venv or system)
- ✅ Checks if already running
- ✅ Error handling

## 🔧 Troubleshooting

**Scripts not appearing?**
- Make sure you added the `raycast-scripts` directory (not individual files)
- Check file permissions: `chmod +x *.sh`
- Restart Raycast

**Scripts not executing?**
- Check Raycast has Accessibility permissions
- System Settings → Privacy & Security → Accessibility → Raycast

**Python not found?**
- Make sure virtual environment exists: `python3 -m venv venv`
- Or ensure system Python is installed

