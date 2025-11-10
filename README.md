# Turn-Taking Prediction Pipeline

**A modular, config-driven pipeline for analyzing turn-taking dynamics in conversational MEG data.**

---

## Overview

This pipeline integrates multimodal data (MEG, audio, transcripts) to predict turn-taking behavior in natural conversations. The analysis combines:

- **Audio-MEG synchronization** with drift correction
- **Prosodic feature extraction** (F0, energy, pauses, speech rate)
- **Linguistic predictability** (surprisal, entropy from LLMs)
- **LLM-derived pragmatic features** (dialogue acts, yield/hold probabilities)
- **Neural features** (ERP, time-frequency, TRF)
- **Hierarchical modeling** (mixed-effects, survival analysis)

---

## Project Status

### Phase 1: Core Infrastructure ✅ (Current)
- [x] Project structure and configuration system
- [x] Audio-MEG synchronization with drift correction
- [ ] ASR and word-level timestamp extraction
- [ ] Prosody feature extraction
- [ ] Basic TRP (Transition Relevance Place) detection
- [ ] QC framework and diagnostics

### Phase 2: Advanced Features (Planned)
- [ ] Linguistic predictability (surprisal/entropy)
- [ ] LLM-derived pragmatic features
- [ ] Enhanced neural feature extraction
- [ ] Ba/Da localiser integration

### Phase 3: Modeling (Planned)
- [ ] Hierarchical logistic regression
- [ ] Survival modeling for latency
- [ ] Engagement moderation/mediation
- [ ] Complete QC and reporting

---

## Installation

### Quick Start (Recommended)

```bash
cd /Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline

# 1. Create virtual environment
make setup-venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install package and dependencies
make install

# 4. Verify installation
python -c "import mne, librosa, whisper; print('All dependencies installed!')"
```

### Manual Installation (if you don't have make)

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install package
pip install --upgrade pip
pip install -e .
```

### Optional: Install development tools
```bash
make install-dev  # Installs pytest, black, jupyter, etc.
```

### Optional: Set up local data directory
To create a self-contained project with symlinks to large files:
```bash
python scripts/setup_data.py
```

This will:
- Copy transcripts into `data/raw/transcripts/`
- Create symlinks to MEG files in `data/raw/meg_links/`
- Generate `data/DATA_MANIFEST.json` for reproducibility

### Alternative: Using Conda (optional)
If you prefer conda:
```bash
conda env create -f environment.yml
conda activate turn-taking-pipeline
```

---

## Configuration

All pipeline parameters are defined in `config/config.yaml`. Key sections:

### Data Paths
```yaml
data:
  meg_base_dir: "/path/to/natural-conversations-bids/derivatives/mne-bids-pipeline-..."
  transcript_dir: "/path/to/audios_metadata_labelled"
  freesurfer_dir: "/path/to/freesurfer/subjects"
```

### Subject Selection
```yaml
subjects:
  include: "all"  # or ["sub-01", "sub-02", ...]
  exclude: []
  subject_range: [1, 32]
```

### Synchronization
```yaml
sync:
  aux_channel_name: "MISC007"
  fs_meg_aux: 1000
  use_windowed_sync: true
  window_size_s: 90
  target_alignment_error_ms: 10.0
```

See `config/config.yaml` for full documentation.

---

## Usage

### Phase 1: Audio-MEG Synchronization

```python
from src.utils.config import load_config, get_subject_paths
from src.utils.io import load_meg_raw
from src.sync import synchronize_audio_meg
from src.utils.logging_setup import setup_logging

# Load configuration
config = load_config()
logger = setup_logging(config["processing"]["log_file"])

# Process a single subject/run
subject = "sub-01"
run = 1  # Conversation block 1

paths = get_subject_paths(subject, run, config)
meg_raw = load_meg_raw(paths["meg_raw"])

# Synchronize (need to get external audio path from your setup)
sync_params = synchronize_audio_meg(
    meg_raw,
    external_audio_path=paths["external_audio"],  # You'll need to add this
    config=config
)

# Convert external timestamps to MEG time
from src.sync import ext_to_meg_time
import pandas as pd

transcript = pd.read_csv(paths["transcript"])
transcript["start_meg"] = ext_to_meg_time(transcript["start"].values, sync_params)
transcript["end_meg"] = ext_to_meg_time(transcript["end"].values, sync_params)
```

### Run Full Pipeline (when complete)
```bash
python scripts/run_pipeline.py --config config/config.yaml --subjects sub-01 sub-02
```

---

## Data Structure

### Input Data
```
External data (OneDrive):
├── MEG files (.fif)
│   └── .../derivatives/mne-bids-pipeline-20240628/sub-XX/meg/
│       ├── sub-XX_task-conversation_run-YY_proc-clean_raw.fif
│       ├── sub-XX_task-conversation_fwd.fif
│       └── sub-XX_task-rest_proc-clean_raw.fif
├── Transcripts (.csv)
│   └── .../audios_metadata_labelled/GXX_BYY.csv
└── FreeSurfer
    └── .../freesurfer/subjects/sub-XX/

Project data (local):
├── data/
│   ├── raw/
│   │   ├── transcripts/  (copied)
│   │   ├── meg_links/    (symlinked)
│   │   └── freesurfer/   (symlinked)
│   └── DATA_MANIFEST.json
```

### Output Structure
```
outputs/
├── sync/                    # Synchronization results
│   └── sub-XX/
│       └── run-YY/
│           ├── sync_params.json
│           ├── envelope_alignment.png
│           └── drift_correction.png
├── features/                # Feature tables
│   └── sub-XX/
│       └── run-YY/
│           ├── prosody_features.csv
│           └── word_timestamps.csv
├── events/                  # TRP events
│   └── sub-XX/
│       └── run-YY/
│           └── trp_events.csv
└── qc/                      # Quality control
    └── reports/
```

---

## Experimental Design

- **32 subjects** (sub-01 through sub-32)
- **6 runs per subject**:
  - Runs 1, 3, 5: Conversation condition
  - Runs 2, 4: Repetition condition
  - Run 6: Ba/Da localiser

### File Naming Conventions
- MEG: `sub-{XX}` (e.g., `sub-01`)
- Transcripts: `G{XX}_B{YY}` (e.g., `G01_B1`)
- Mapping: `sub-01` = `G01`, run 1 = `B1`

---

## Key Modules

### `src/sync/`
- **`audio_meg_sync.py`**: Main synchronization logic
  - Cross-correlation-based offset estimation
  - Drift correction with windowed analysis
  - QC metrics (median alignment error, drift rate)

- **`drift_correction.py`**: Time-varying drift estimation
  - Sliding window cross-correlation
  - Linear/piecewise drift models

### `src/features/` (Phase 1, in progress)
- **`prosody.py`**: Prosodic feature extraction
- **`linguistic.py`**: Surprisal and entropy (Phase 2)
- **`llm_pragmatic.py`**: LLM-derived features (Phase 2)

### `src/utils/`
- **`config.py`**: Configuration loading and path management
- **`io.py`**: Data loading/saving utilities
- **`logging_setup.py`**: Logging configuration

---

## Development Roadmap

### Immediate Next Steps (Phase 1 Completion)
1. ✅ Synchronization module
2. ASR integration (Whisper)
3. Prosody feature extraction
4. TRP detection
5. Integration testing on 3 subjects

### Phase 2 (Advanced Features)
- Transformer-based linguistic predictability
- LLM dialogue act classification
- Semantic embedding projections

### Phase 3 (Modeling)
- Bayesian hierarchical models
- Survival analysis for turn latency
- Cross-validation framework

---

## Quality Control

### Synchronization QC
- **Target**: Median alignment error < 10 ms
- **Accept**: < 50 ms
- **Metrics**:
  - Envelope alignment plots
  - Onset detection validation
  - Drift rate (ppm)

### Feature QC
- Distribution plots
- Missing value checks
- VIF for multicollinearity
- Outlier detection (>5 SD)

---

## Citation

If you use this pipeline, please cite:
```bibtex
@software{turn_taking_pipeline,
  title={Turn-Taking Prediction Pipeline},
  author={Natural Conversations Study Team},
  year={2025},
  institution={AUT University}
}
```

---

## Contact

For questions or issues, please contact the Natural Conversations Study team.

---

## License

[Add license information]
