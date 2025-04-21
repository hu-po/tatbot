#!/bin/bash

# copy config files to the UI directory
bash ~/dev/tatbot-dev/scripts/trossen-ai/copy-ui-config.sh

# clear out old recordings
bash ~/dev/tatbot-dev/scripts/trossen-ai/clear-hf-cache.sh

echo "Starting Trossen AI Data Collection UI..."
DISPLAY=:0 gtk-launch trossen_ai_data_collection_ui