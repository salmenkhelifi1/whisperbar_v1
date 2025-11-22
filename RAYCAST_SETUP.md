# Raycast Setup for WhisperBar

## Quick Setup - Two Commands

### 1. Start WhisperBar

**Create Script Command in Raycast:**
- Title: `WhisperBar Start`
- Subtitle: `Launch Speech-to-Text App`
- Script Path: `/Users/salmenkhelifi/Documents/whisperbar_v1/whisperbar_start.sh`
- Mode: `Run in Background`

**Or use inline script:**
```bash
cd /Users/salmenkhelifi/Documents/whisperbar_v1
if pgrep -f "python.*main.py" > /dev/null; then
    echo "Already running"
else
    if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
        nohup venv/bin/python main.py > /tmp/whisperbar.log 2>&1 &
    elif command -v python3 &> /dev/null; then
        nohup python3 main.py > /tmp/whisperbar.log 2>&1 &
    else
        nohup python main.py > /tmp/whisperbar.log 2>&1 &
    fi
    sleep 2 && echo "✅ Started"
fi
```

---

### 2. Stop WhisperBar

**Create Script Command in Raycast:**
- Title: `WhisperBar Stop`
- Subtitle: `Stop Speech-to-Text App`
- Script Path: `/Users/salmenkhelifi/Documents/whisperbar_v1/whisperbar_stop.sh`
- Mode: `Run in Background`

**Or use inline script:**
```bash
if pgrep -f "python.*main.py" > /dev/null; then
    pkill -f "python.*main.py" && echo "✅ Stopped"
else
    echo "Not running"
fi
```

---

## One-Liner Versions (Copy-Paste Ready)

### Start (Inline Script):
```bash
cd /Users/salmenkhelifi/Documents/whisperbar_v1 && ([ -d "venv" ] && nohup venv/bin/python main.py > /tmp/whisperbar.log 2>&1 & || nohup python3 main.py > /tmp/whisperbar.log 2>&1 &) && sleep 1 && echo "✅ Started"
```

### Stop (Inline Script):
```bash
pkill -f "python.*main.py" && echo "✅ Stopped" || echo "Not running"
```

---

## Usage

1. **Start:** Open Raycast → Type `whisper start` → Enter
2. **Stop:** Open Raycast → Type `whisper stop` → Enter

Look for "🎤 Mic" icon in menu bar when running.

---

## Troubleshooting

**Check if running:**
```bash
pgrep -f "python.*main.py"
```

**View logs:**
```bash
tail -f /tmp/whisperbar.log
```

**Force stop:**
```bash
pkill -9 -f "python.*main.py"
```

