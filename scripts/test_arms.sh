#!/bin/bash
echo "🦾  Testing arms"
source ~/tatbot/scripts/setup_env.sh
uv pip install .[bot]
uv run -m tatbot.bot.trossen_config
echo "✅ Done"