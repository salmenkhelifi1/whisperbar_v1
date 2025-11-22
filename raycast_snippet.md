# Raycast Integration for WhisperBar

## Quick Setup

### Option 1: Shell Script (Recommended)

1. **Create a Raycast Script Command:**
   - Open Raycast
   - Go to Extensions → Script Commands → Create Script
   - Choose "Shell Script"
   - Set the following:

**Title:** `WhisperBar`
**Subtitle:** `Launch WhisperBar Speech-to-Text`
**Script Path:** `/Users/salmenkhelifi/Documents/whisperbar_v1/run_whisperbar.sh`
**Mode:** `Run in Background`

**Script:**
```bash
#!/bin/bash

# WhisperBar Launcher Script for Raycast
SCRIPT_DIR="/Users/salmenkhelifi/Documents/whisperbar_v1"
cd "$SCRIPT_DIR"

# Activate virtual environment and run
source venv/bin/activate && python main.py
```

### Option 2: Direct Python Command

**Title:** `WhisperBar`
**Subtitle:** `Launch WhisperBar Speech-to-Text`
**Script:** `Inline Script`
**Mode:** `Run in Background`

**Script:**
```bash
#!/bin/bash

cd /Users/salmenkhelifi/Documents/whisperbar_v1
source venv/bin/activate
python main.py
```

### Option 3: AppleScript (Alternative)

**Title:** `WhisperBar`
**Subtitle:** `Launch WhisperBar Speech-to-Text`
**Script:** `AppleScript`
**Mode:** `Run in Background`

**Script:**
```applescript
tell application "Terminal"
    do script "cd /Users/salmenkhelifi/Documents/whisperbar_v1 && source venv/bin/activate && python main.py"
    activate
end tell
```

---

## Raycast Configuration

### Recommended Settings:

- **Icon:** Use a microphone emoji 🎤 or custom icon
- **Keywords:** `whisper`, `speech`, `transcribe`, `voice`
- **Mode:** `Run in Background` (so it doesn't block Raycast)
- **Argument:** None (or optional: model name)

---

## Advanced: With Arguments

If you want to pass arguments (e.g., model name):

**Script:**
```bash
#!/bin/bash

cd /Users/salmenkhelifi/Documents/whisperbar_v1
source venv/bin/activate

# Optional: Pass model name as argument
if [ -n "$1" ]; then
    python main.py --model "$1"
else
    python main.py
fi
```

**Argument:** `{Query}` (optional)

---

## Quick Install Script

Save this as `setup_raycast.sh` and run it:

```bash
#!/bin/bash

RAYCAST_SCRIPT_DIR="$HOME/Library/Application Support/Raycast/Scripts"
SCRIPT_NAME="whisperbar.sh"

mkdir -p "$RAYCAST_SCRIPT_DIR"

cat > "$RAYCAST_SCRIPT_DIR/$SCRIPT_NAME" << 'EOF'
#!/bin/bash
cd /Users/salmenkhelifi/Documents/whisperbar_v1
source venv/bin/activate
python main.py
EOF

chmod +x "$RAYCAST_SCRIPT_DIR/$SCRIPT_NAME"
echo "✅ Raycast script created at: $RAYCAST_SCRIPT_DIR/$SCRIPT_NAME"
echo "Now add it in Raycast: Extensions → Script Commands → Import"
```

---

## Usage

Once set up:
1. Open Raycast (`Cmd+Space`)
2. Type `whisper` or `speech`
3. Press Enter
4. WhisperBar will launch in the background
5. Look for "🎤 Mic" icon in menu bar

---

## Troubleshooting

**Script not executable?**
```bash
chmod +x /Users/salmenkhelifi/Documents/whisperbar_v1/run_whisperbar.sh
```

**Virtual environment not found?**
- Make sure you've created it: `python3 -m venv venv`
- Install dependencies: `source venv/bin/activate && pip install -r requirements.txt`

**Permission denied?**
- Check Raycast has Accessibility permissions
- Check Terminal/iTerm has Accessibility permissions

---

## One-Liner for Raycast

**Simplest version** (paste directly in Raycast Script Command):

```bash
cd /Users/salmenkhelifi/Documents/whisperbar_v1 && source venv/bin/activate && python main.py &
```

The `&` runs it in background so Raycast doesn't wait.

