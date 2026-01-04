#!/bin/bash
# Resume GPT-OSS-20B download
# This runs in background and just downloads model files (doesn't load into GPU)

cd /home/poss/Documents/Code/HatCatDev

echo "============================================================"
echo "  Resuming GPT-OSS-20B Download"
echo "============================================================"
echo "Started: $(date)"
echo ""

# Use Python to trigger download via HuggingFace
.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os

print('Downloading openai/gpt-oss-20b...')
print('This will resume from where it left off.')
print('')

try:
    path = snapshot_download(
        'openai/gpt-oss-20b',
        resume_download=True,
        max_workers=2,  # Limit bandwidth usage
    )
    print(f'Download complete! Model cached at: {path}')
except Exception as e:
    print(f'Download error: {e}')
"

echo ""
echo "============================================================"
echo "  Download Finished"
echo "============================================================"
echo "Finished: $(date)"
