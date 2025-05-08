#!/bin/bash
echo "Copying UI config files..."
cp ~/dev/tatbot-dev/config/trossen-ui/tasks.yaml /home/trossen-ai/./miniconda3/envs/trossen_ai_data_collection_ui_env/lib/python3.10/site-packages/trossen_ai_data_collection_ui/configs/tasks.yaml
cp ~/dev/tatbot-dev/config/trossen-ui/trossen_ai_robots.yaml /home/trossen-ai/./miniconda3/envs/trossen_ai_data_collection_ui_env/lib/python3.10/site-packages/trossen_ai_data_collection_ui/configs/robot/trossen_ai_robots.yaml
echo "Done!"