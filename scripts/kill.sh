#!/bin/bash
echo "☠️  Killing all tatbot processes"
sudo pkill -9 -f tatbot || true
echo "✅ Done"