#!/bin/bash
# Double-click this file in Finder to start the booth.
# Leave this window open while the booth is running -- closing it stops
# everything. To stop on purpose, click into this window and press
# Control-C, or just close the window.

cd "$(dirname "$0")"

echo "Starting the booth. Leave this window open."
echo "To stop: press Control-C, or close this window."
echo ""

.venv/bin/python convention_runner.py

echo ""
echo "The booth has stopped."
read -p "Press Enter to close this window..."
