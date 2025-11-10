#!/bin/bash
# Simple runner for the sync test

cd "$(dirname "$0")"

# Activate venv
source venv/bin/activate

# Run test
python scripts/test_sync_complete.py "$@"
