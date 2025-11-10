# Installation Guide

## Simple Installation (No Conda!)

### Prerequisites
- Python 3.9 or higher
- pip (comes with Python)
- `make` (optional, for convenience)

### Step-by-Step

```bash
# 1. Navigate to project
cd /Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install package and all dependencies
pip install -e .

# 6. Verify installation
python -c "import mne, librosa, whisper; print('✓ All dependencies installed!')"
```

### Using Makefile (Shortcut)

If you have `make` installed:

```bash
# One command to create venv
make setup-venv

# Activate it
source venv/bin/activate

# Install everything
make install
```

### Install with Development Tools

For testing, formatting, and Jupyter notebooks:

```bash
pip install -e ".[dev]"
```

Or with make:
```bash
make install-dev
```

## What Gets Installed

### Core Dependencies
- **MNE-Python** (≥1.5.0) - MEG/EEG processing
- **Librosa** (≥0.10.0) - Audio processing
- **SoundFile** (≥0.12.0) - Audio I/O
- **Parselmouth** (≥0.4.3) - Prosody extraction (Praat)
- **Whisper** (≥20230314) - Speech recognition
- **NumPy** (≥1.24.0) - Numerical computing
- **SciPy** (≥1.11.0) - Signal processing
- **Pandas** (≥2.0.0) - Data manipulation
- **PyYAML** (≥6.0) - Config files
- **Matplotlib** (≥3.7.0) - Plotting
- **Seaborn** (≥0.12.0) - Statistical visualization
- **Scikit-learn** (≥1.3.0) - Machine learning
- **Statsmodels** (≥0.14.0) - Statistical models

### Optional: Development Tools
```bash
pip install -e ".[dev]"
```
Adds: pytest, black, flake8, jupyter, ipython

### Optional: LLM Features (Phase 2)
```bash
pip install -e ".[llm]"
```
Adds: torch, transformers, sentence-transformers

### Optional: Advanced Modeling (Phase 3)
```bash
pip install -e ".[modeling]"
```
Adds: bambi, pymc, lifelines, arviz

## Troubleshooting

### Whisper Installation Issues

If Whisper fails to install:
```bash
# Install ffmpeg first (required for Whisper)
# On macOS:
brew install ffmpeg

# On Linux:
sudo apt-get install ffmpeg

# Then retry:
pip install openai-whisper
```

### MNE Installation Issues

If MNE has problems:
```bash
# Try installing from conda-forge first
pip install mne[hdf5]
```

### M1/M2 Mac Issues

On Apple Silicon Macs, some packages may need special handling:
```bash
# Use miniforge instead of standard conda
# Or ensure you're using native arm64 Python:
python3 --version  # Should show arm64
```

## Verification

After installation, verify everything works:

```python
import mne
import librosa
import whisper
import pandas as pd
import numpy as np
from scipy import signal
import parselmouth

print("✓ All imports successful!")
print(f"MNE version: {mne.__version__}")
print(f"Librosa version: {librosa.__version__}")
```

## Deactivating

When you're done working:
```bash
deactivate
```

## Uninstalling

To completely remove:
```bash
# Deactivate first
deactivate

# Remove virtual environment
rm -rf venv/

# Remove package build artifacts
make clean  # or manually: rm -rf build/ dist/ *.egg-info
```

## Alternative: Global Installation (Not Recommended)

If you really want to install globally (not recommended):
```bash
pip install .  # Without -e flag
```

But **always use virtual environments** for project isolation!
