#!/bin/bash
echo "☠️  Killing all tatbot processes"
pkill -9 -f tatbot || true
echo "✅ Done"