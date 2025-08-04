#!/bin/bash
echo "☠️  Killing all tatbot related processes"
pkill -9 -f "tatbot.mcp" || true
echo "✅ Done"