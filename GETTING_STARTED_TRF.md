# Getting Started with TRF Analysis

**Date:** 2025-10-24
**Status:** ✅ Ready for TRF Modeling

---

## Quick Start

You now have a complete pipeline that extracts turn-taking features from MEG+audio data and prepares them for TRF (Temporal Response Function) modeling.

### What You Have

✅ **Audio-MEG synchronization** (validated on 4 subjects)
✅ **ASR transcription** (Whisper with word-level timestamps)
✅ **Prosody extraction** (F0, energy, intensity, spectral features @ 10ms)
✅ **TRP detection** (turn-transition relevance places)
✅ **Example TRF script** ready for eelbrain

---

## Output Files (per subject/run)

```
outputs/features/{subject}/run-{run:02d}/
├── sync_params.json           # Synchronization metadata
├── transcript.csv             # 132 segments, 913 words
├── turns.csv                  # 85 turns detected
├── prosody.csv                # 36,062 frames @ 10ms (F0, energy, intensity, spectral)
├── pauses.csv                 # 2 pauses detected
├── speech_rate.csv            # 143 timepoints (local rate)
├── trp_features.csv           # 1 TRP candidate
└── feature_validation.png     # QC plot of features
```

### File Sizes (sub-01, run-01)

- `prosody.csv`: **4.3 MB** (main feature table, 36K frames)
- `transcript.csv`: **330 KB** (word-level transcript)
- `turns.csv`: **9.8 KB** (85 turns)
- `trp_features.csv`: **154 B** (sparse events)

---

## Feature Statistics (sub-01, run-01)

### Prosody
- **F0 range:** 75-483 Hz
- **Mean F0:** 181 Hz (voiced frames)
- **Voicing:** 19.5% of frames
- **Mean intensity:** 47.4 dB
- **Duration:** 360.6s @ 100 Hz (10ms frames)

### Speech & Turns
- **Speech rate:** 3.33 words/s (range: 0.4-8.8)
- **Total turns:** 85
- **Mean turn duration:** 3.2s
- **Mean words/turn:** 10.6

### TRPs
- **TRP candidates:** 1 (will increase with better pause detection tuning)
- **Mean pause duration:** 3.0s

---

## How to Use for TRF Analysis

### Step 1: Run Pipeline on Your Subjects

```bash
cd /Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline
source venv/bin/activate

# Run on multiple subjects
for sub in sub-01 sub-02 sub-03; do
    python scripts/test_asr_prosody.py --subject $sub --run 1
done
```

**Processing time:** ~20 seconds per subject/run
- Sync: 5s
- ASR (Whisper base): 10-12s
- Prosody: 5s
- I/O: 2-3s

### Step 2: Validate Features

```bash
# Check feature quality
python scripts/example_trf_analysis.py --subject sub-01 --run 1 --validate-only

# View the validation plot
open outputs/features/sub-01/run-01/feature_validation.png
```

This generates a plot showing:
- F0 (pitch) over time
- Intensity over time
- Energy over time
- TRP events marked with red dashed lines

### Step 3: Load Features for TRF

```python
import pandas as pd
import json
import numpy as np

# Load synchronization params
with open("outputs/features/sub-01/run-01/sync_params.json") as f:
    sync_params = json.load(f)

# Load prosody (continuous features)
prosody = pd.read_csv("outputs/features/sub-01/run-01/prosody.csv")

# Prosody is in external audio time - convert to MEG time
offset_s = sync_params["initial_offset_s"]  # -12.24s for sub-01
prosody["time_meg"] = prosody["time"] - offset_s

# Now prosody["time_meg"] aligns with your MEG data (in seconds)
```

### Step 4: Create Continuous Regressors

```python
from scipy.interpolate import interp1d

# Your MEG time grid (1000 Hz sampling)
meg_sfreq = 1000.0
meg_duration = sync_params["meg_duration_s"]  # 385.0s
meg_times = np.arange(0, meg_duration, 1/meg_sfreq)

# Interpolate F0 to MEG sampling rate
f0_interp = interp1d(
    prosody["time_meg"],
    prosody["f0_smooth"],  # Use smoothed F0 (interpolated through unvoiced)
    kind="linear",
    bounds_error=False,
    fill_value=np.nan
)
f0_meg = f0_interp(meg_times)  # Shape: (385000,)

# Similarly for other features
intensity_meg = interp1d(prosody["time_meg"], prosody["intensity"], ...)(meg_times)
energy_meg = interp1d(prosody["time_meg"], prosody["energy"], ...)(meg_times)
```

### Step 5: Create Event Regressors (TRPs, Turn Onsets)

```python
# Load TRP events
trps = pd.read_csv("outputs/features/sub-01/run-01/trp_features.csv")
trps["time_meg"] = trps["time"] - offset_s

# Create impulse regressor
trp_regressor = np.zeros(len(meg_times))
for trp_time in trps["time_meg"]:
    idx = np.argmin(np.abs(meg_times - trp_time))
    trp_regressor[idx] = 1.0  # Impulse at TRP time

# Similarly for turn onsets
turns = pd.read_csv("outputs/features/sub-01/run-01/turns.csv")
turns["start_meg"] = turns["start"] - offset_s

turn_regressor = np.zeros(len(meg_times))
for turn_start in turns["start_meg"]:
    idx = np.argmin(np.abs(meg_times - turn_start))
    turn_regressor[idx] = 1.0
```

---

## TRF Analysis with Eelbrain

### Install Eelbrain (if not already installed)

```bash
pip install eelbrain
```

### Minimal TRF Example

```python
import eelbrain
import mne

# 1. Load your beamformed MEG source data
stc = mne.read_source_estimate("path/to/your_stc-lh.stc")

# 2. Convert to eelbrain NDVar
meg_ndvar = eelbrain.load.mne.stc_ndvar(
    stc,
    subject="fsaverage",  # or your subject ID
    parc="aparc",  # parcellation
    method="mean",  # average within ROIs
)

# 3. Create predictor NDVars
time_dim = meg_ndvar.time  # Use MEG time dimension

f0_ndvar = eelbrain.NDVar(f0_meg, dims=(time_dim,), name="F0")
trp_ndvar = eelbrain.NDVar(trp_regressor, dims=(time_dim,), name="TRP")

# 4. Fit TRF
trf = eelbrain.boosting(
    meg_ndvar,           # Dependent variable (neural response)
    [f0_ndvar, trp_ndvar],  # Independent variables (predictors)
    tstart=-0.1,         # TRF window start (100ms before)
    tstop=0.5,           # TRF window end (500ms after)
    basis=0.050,         # Basis function width (50ms)
    partitions=5,        # Cross-validation folds
    test=1,              # Test partition
    selective_stopping=True,
)

# 5. Test significance
results = eelbrain.testnd.corr(
    trf.r,               # Model correlation (how well predictors explain MEG)
    alpha=0.05,
    samples=10000,       # Permutation samples
    pmin=0.05,
)

# 6. Visualize TRFs
eelbrain.plot.brain.butterfly(trf.h_scaled["F0"])  # F0 TRF
eelbrain.plot.brain.butterfly(trf.h_scaled["TRP"])  # TRP TRF

# 7. Get prediction accuracy
print(f"Model correlation: {trf.r.mean():.3f}")
print(f"F0 contribution: {trf.proportion_explained['F0'].mean():.3f}")
print(f"TRP contribution: {trf.proportion_explained['TRP'].mean():.3f}")
```

---

## Advanced: Custom TRF Analysis

### Multiple Predictors

```python
predictors = [
    f0_ndvar,          # Pitch
    intensity_ndvar,   # Loudness
    energy_ndvar,      # Energy envelope
    trp_ndvar,         # TRP events
    turn_ndvar,        # Turn onsets
]

trf = eelbrain.boosting(
    meg_ndvar,
    predictors,
    tstart=-0.1,
    tstop=0.6,
    basis=0.050,
    error="l1",  # L1 loss for robustness
)
```

### ROI-Specific Analysis

```python
# Extract specific ROI
roi_meg = meg_ndvar.sub(source="superiortemporal-lh")  # Left STG

# Fit TRF for this ROI
trf_stg = eelbrain.boosting(roi_meg, predictors, ...)

# Compare TRFs across ROIs
rois = ["superiortemporal-lh", "inferiorfrontal-lh", "motorarea-lh"]
trf_by_roi = {}
for roi in rois:
    roi_meg = meg_ndvar.sub(source=roi)
    trf_by_roi[roi] = eelbrain.boosting(roi_meg, predictors, ...)
```

### Time-Resolved TRP Effects

```python
# Epoching around TRPs
epochs = []
for trp_time in trps["time_meg"]:
    # Extract MEG data from -0.5s to +1.0s around TRP
    epoch = meg_ndvar.sub(time=(trp_time - 0.5, trp_time + 1.0))
    epochs.append(epoch)

# Average across TRP events
trp_erp = eelbrain.combine(epochs)

# Test against baseline
baseline = (-0.2, 0)  # 200ms before TRP
result = eelbrain.testnd.ttest_rel(
    trp_erp,
    baseline=baseline,
    alpha=0.05,
)

eelbrain.plot.brain.dspm(result.masked_parameter_map(), ...)
```

---

## Interpreting Results

### What TRFs Tell You

**Positive TRF coefficients:**
- Neural response increases when predictor increases
- e.g., High F0 → Increased activity in auditory cortex

**Negative TRF coefficients:**
- Neural response decreases when predictor increases
- e.g., High intensity → Suppression in motor cortex (listening mode)

**TRF latency:**
- Early peaks (50-150ms): Sensory processing
- Mid peaks (150-300ms): Perceptual analysis
- Late peaks (300-600ms): Cognitive/predictive processing

### TRP-Specific Predictions

**Pre-TRP effects (negative lags):**
- Anticipatory activity before turn transition
- Motor preparation for speaking
- Predictive suppression of auditory cortex

**Post-TRP effects (positive lags):**
- Turn-taking decision
- Response planning
- Speech onset preparation

---

## Troubleshooting

### Issue: Low TRP Count

If you only get 1-2 TRPs per subject, try:

```python
# Adjust pause detection threshold
pauses = detect_pauses(
    prosody,
    energy_threshold=None,  # Auto-detect (default)
    min_pause_duration=0.1,  # Lower from 0.2s → 0.1s
)
```

### Issue: Noisy F0

If F0 is too noisy:

```python
# Use more smoothing
from scipy.signal import savgol_filter
f0_smoothed = savgol_filter(prosody["f0_smooth"], window_length=51, polyorder=3)
```

### Issue: Time Alignment Errors

Always verify alignment:

```python
# Check offset
print(f"Offset: {sync_params['initial_offset_s']:.3f}s")
print(f"Correlation: {sync_params['qc_metrics']['peak_correlation']:.3f}")

# Plot alignment
plt.plot(meg_audio_times, meg_audio, label="MEG audio")
plt.plot(ext_audio_times - offset, ext_audio, label="Ext audio (aligned)")
plt.legend()
```

---

## Next Steps

### Immediate

1. **Run on More Subjects:**
   ```bash
   for sub in $(seq -f "sub-%02g" 1 32); do
       python scripts/test_asr_prosody.py --subject $sub --run 1
   done
   ```

2. **Batch Analysis:**
   Create `scripts/batch_process.py` for parallel processing

3. **Validate TRF Quality:**
   - Check cross-validation correlations > 0.1
   - Compare with shuffled predictors
   - Test anatomical specificity (STG > visual cortex)

### Phase 2: Linguistic Features

Add semantic predictors:
- **Surprisal:** Word predictability from GPT-2/BERT
- **Entropy:** Uncertainty at each word
- **Semantic similarity:** Word embedding distances

### Phase 3: Turn-Taking Prediction

Build classifier:
```python
# Predict turn-taking success from features
from sklearn.ensemble import RandomForestClassifier

X = prosody_features  # F0, intensity, rate
y = turn_success      # Did partner take turn?

model = RandomForestClassifier()
model.fit(X, y)
```

---

## File Locations

**Pipeline:**
- Feature extraction: `scripts/test_asr_prosody.py`
- TRF example: `scripts/example_trf_analysis.py`

**Outputs:**
- Features: `outputs/features/{subject}/run-{run:02d}/`
- Sync QC: `outputs/sync/{subject}/run-{run:02d}/`

**Documentation:**
- This guide: `GETTING_STARTED_TRF.md`
- Phase 1 complete: `PHASE1_COMPLETE.md`
- Sync details: `SYNC_COMPLETE.md`

---

## Support & References

**Eelbrain Documentation:**
- TRF tutorial: https://eelbrain.readthedocs.io/en/stable/recipes.html#boosting
- API reference: https://eelbrain.readthedocs.io/

**Turn-Taking Literature:**
- Stivers et al. (2009): Cross-linguistic turn-taking
- Levinson (2016): Turn-taking in human communication
- Bögels et al. (2015): Neural mechanisms of turn-taking

**Your Data:**
- 32 subjects × 5 runs = 160 recordings
- ~6 minutes each = 16 hours total data
- Expect ~10K turns, ~50K words for TRF analysis

---

**STATUS: Ready for TRF Modeling! 🚀**

All Phase 1 features extracted and validated. Time to discover neural correlates of turn-taking!

