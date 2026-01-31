#!/bin/bash

# Setup cron job for Zerodha auto login script
# This script configures a cron job to run at 4:13 PM on weekdays (Monday-Friday)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/zerodha_auto_login_nse.py"
PYTHON_PATH="/usr/bin/python3"

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found at $SCRIPT_PATH"
    exit 1
fi

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python not found at $PYTHON_PATH"
    echo "Please update PYTHON_PATH in this script to point to your Python 3 executable"
    exit 1
fi

# Create cron entry
CRON_ENTRY="13 16 * * 1-5 $PYTHON_PATH $SCRIPT_PATH"

# Check if cron entry already exists
if crontab -l 2>/dev/null | grep -q "$SCRIPT_PATH"; then
    echo "Cron entry already exists for this script."
    echo "Current cron entries:"
    crontab -l 2>/dev/null | grep "$SCRIPT_PATH"
    echo ""
    read -p "Do you want to remove the existing entry and add a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove existing entry
        crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH" | crontab -
        echo "Removed existing cron entry."
    else
        echo "Keeping existing cron entry. Exiting."
        exit 0
    fi
fi

# Add new cron entry
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

if [ $? -eq 0 ]; then
    echo "Cron job added successfully!"
    echo ""
    echo "Cron entry: $CRON_ENTRY"
    echo ""
    echo "This will run the script at 4:00 PM (16:13) on weekdays (Monday-Friday)."
    echo ""
    echo "To view your cron jobs, run: crontab -l"
    echo "To remove this cron job, run: crontab -e"
else
    echo "Error: Failed to add cron job."
    exit 1
fi

