# 📋 Step-by-Step Installation Instructions

## Add Script Directory to Raycast

Follow these exact steps:

### Step 1: Open Raycast
- Press `Cmd + Space` (or your Raycast hotkey)
- This opens the Raycast command palette

### Step 2: Open Extensions
- Type: `Extensions`
- Press `Enter`
- This opens the Extensions/Preferences window

### Step 3: Add Script Directory
- Click the **`+`** button (usually in the top-right or bottom-left)
- Click **`Add Script Directory`** from the dropdown menu
- A file picker dialog will open

### Step 4: Select the Directory
- Navigate to: `/Users/salmenkhelifi/Documents/whisperbar_v1/`
- Select the **`raycast-scripts`** folder
- Click **`Open`** or **`Select`**

### Step 5: Verify Installation
- You should see "raycast-scripts" appear in your Script Commands list
- Close the Extensions window

### Step 6: Test It!
- Press `Cmd + Space` to open Raycast
- Type: `WhisperBar` or `whisper`
- You should see three commands:
  - 🎤 **WhisperBar Start**
  - 🛑 **WhisperBar Stop**
  - 🔄 **WhisperBar Restart**

## ✅ Quick Path Reference

**Full path to select:**
```
/Users/salmenkhelifi/Documents/whisperbar_v1/raycast-scripts
```

**Or navigate manually:**
1. Open Finder
2. Go to: Documents → whisperbar_v1 → raycast-scripts
3. Copy this path and paste it in Raycast's file picker

## 🎯 Visual Guide

```
Raycast (Cmd+Space)
  └─> Type "Extensions" → Enter
      └─> Click "+" button
          └─> "Add Script Directory"
              └─> Navigate to: /Users/salmenkhelifi/Documents/whisperbar_v1/raycast-scripts
                  └─> Click "Open"
                      └─> Done! ✅
```

## 🔍 Troubleshooting

**Can't find the "+" button?**
- Make sure you're in the Extensions/Preferences window
- Look for "Script Commands" section
- The "+" might be at the bottom of the Script Commands list

**Directory not showing scripts?**
- Make sure you selected the `raycast-scripts` folder (not the parent folder)
- Verify files exist: `ls /Users/salmenkhelifi/Documents/whisperbar_v1/raycast-scripts/`
- Restart Raycast if needed

**Scripts not appearing?**
- Wait a few seconds for Raycast to index them
- Try typing "WhisperBar" in Raycast search
- Check Raycast → Extensions → Script Commands to see if the directory is listed

