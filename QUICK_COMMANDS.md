# WhisperBar - Quick Start/Stop/Restart Guide

## 🚀 Quick Commands

### Using the Main Script (Recommended)

Navigate to the project directory first:
```bash
cd /Users/salmenkhelifi/Documents/whisperbar_v1
```

**Start WhisperBar:**
```bash
./run_whisperbar.sh start
```

**Stop WhisperBar:**
```bash
./run_whisperbar.sh stop
```

**Restart WhisperBar:**
```bash
./run_whisperbar.sh restart
```

**Check Status:**
```bash
./run_whisperbar.sh status
```

---

### Alternative Methods

#### Method 1: Using Individual Scripts

**Start:**
```bash
./whisperbar_start.sh
```

**Stop:**
```bash
./whisperbar_stop.sh
```

#### Method 2: Using macOS Command File

Double-click `WhisperBar.command` in Finder, or run:
```bash
open WhisperBar.command
```

#### Method 3: Direct Python Command

**Start:**
```bash
cd /Users/salmenkhelifi/Documents/whisperbar_v1
source venv/bin/activate
python main.py &
```

**Stop:**
```bash
pkill -f "python.*main.py"
```

---

## 📋 Status Indicators

- **Menu Bar Icon**: Look for 🎤 icon in your macOS menu bar
- **Check if Running**: `./run_whisperbar.sh status`
- **View Logs**: `tail -f /tmp/whisperbar.log`

---

## 🔧 Troubleshooting

**If the script says "Permission denied":**
```bash
chmod +x run_whisperbar.sh
chmod +x whisperbar_start.sh
chmod +x whisperbar_stop.sh
```

**If WhisperBar won't start:**
```bash
# Check logs
cat /tmp/whisperbar.log

# Check if already running
./run_whisperbar.sh status
```

**Force stop (if normal stop doesn't work):**
```bash
pkill -9 -f "python.*main.py"
```

---

## 💡 Pro Tips

1. **Add to PATH**: You can add the project directory to your PATH for easier access
2. **Create Alias**: Add to `~/.zshrc`:
   ```bash
   alias whisperbar-start='cd /Users/salmenkhelifi/Documents/whisperbar_v1 && ./run_whisperbar.sh start'
   alias whisperbar-stop='cd /Users/salmenkhelifi/Documents/whisperbar_v1 && ./run_whisperbar.sh stop'
   alias whisperbar-restart='cd /Users/salmenkhelifi/Documents/whisperbar_v1 && ./run_whisperbar.sh restart'
   ```

3. **Auto-start on Login**: Add `WhisperBar.command` to System Settings > General > Login Items

