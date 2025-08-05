#!/bin/bash
echo "🌐 Starting chrome browser for viz"
pkill -9 -f chromium 
rm -rf /tmp/viz.log || true 
export DISPLAY=:0
setsid chromium-browser --kiosk http://localhost:8080 --disable-gpu >> /tmp/viz.log 2>&1
echo "✅ Done"