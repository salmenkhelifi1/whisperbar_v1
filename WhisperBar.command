#!/bin/bash

# WhisperBar Launcher (.command file for double-click launching)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Clear the terminal screen
clear

# Launch the QuickStart script
exec ./QuickStart.sh 