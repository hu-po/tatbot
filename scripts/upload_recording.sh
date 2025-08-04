#!/bin/bash
DATASET_NAME="$1"
echo "📦🤗 Uploading recording to huggingface: ${DATASET_NAME}"
huggingface-cli upload tatbot/${DATASET_NAME} ~/tatbot/nfs/recordings/${DATASET_NAME}/ --repo-type dataset
echo "✅ Done"