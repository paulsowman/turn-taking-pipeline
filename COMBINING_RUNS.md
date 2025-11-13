# Combining Runs for Condition-Based Analysis

**Date:** 2025-11-13
**Status:** ✅ Ready

---

## Experimental Design

Your study has two conditions across 5 runs per subject:

### Condition 1: Natural Conversation (Runs 1, 3, 5)
- **Type**: Free conversation
- **Turn-taking**: Spontaneous, unpredictable
- **Neural interest**: Anticipatory signals, prediction, natural turn boundaries

### Condition 2: Nursery Rhyme (Runs 2, 4)
- **Type**: Structured repetition
- **Turn-taking**: Predictable, interviewer speaks → participant repeats
- **Neural interest**: Control for turn structure without spontaneous prediction

---

## Why Concatenate with Gaps?

### Problem: Edge Artifacts

When you concatenate runs directly, you create **artificial turn boundaries** at concatenation points:

```
Run 1 ends: "...and that's what I think."
                                           ↓ ARTIFICIAL BOUNDARY
Run 3 starts: "So do you have any plans..."
```

This creates:
- **Spurious turn transitions** that never actually occurred
- **False predictive signals** if analyzing anticipatory activity
- **Contaminated TRF estimates** near boundaries

### Solution: BAD Annotations

The script adds **2-second BAD annotations** around each concatenation boundary:

```
Run 1: [===========================]
                                    [--BAD--]  ← 2s gap
Run 3:                                        [===========================]
                                              [--BAD--]
Run 5:                                                  [==================]
```

**Result**: TRF analysis automatically excludes these regions, preventing edge artifacts.

---

## Usage

### Basic: Combine All Subjects, Both Conditions

```bash
python scripts/combine_runs.py --all
```

**Output** (per subject):
- `outputs/trf_combined/sub-01/sub-01_conversation_trf_raw.fif`
- `outputs/trf_combined/sub-01/sub-01_nursery_rhyme_trf_raw.fif`
- Metadata JSON files

### Single Subject

```bash
python scripts/combine_runs.py --subject sub-01
```

### Specific Condition Only

```bash
# Only conversation (runs 1, 3, 5)
python scripts/combine_runs.py --subject sub-01 --condition conversation

# Only nursery rhyme (runs 2, 4)
python scripts/combine_runs.py --subject sub-01 --condition nursery_rhyme
```

### Custom Gap Duration

```bash
# Use 3-second gaps instead of default 2s
python scripts/combine_runs.py --subject sub-01 --gap 3.0
```

---

## Output Files

### Combined FIF File

```
outputs/trf_combined/sub-01/sub-01_conversation_trf_raw.fif
```

Contains:
- All MEG channels (306 for Neuromag)
- All 13 TRF predictor channels (MISC)
- BAD annotations at concatenation boundaries
- Continuous data from runs 1 + 3 + 5

### Metadata JSON

```json
{
  "subject": "sub-01",
  "condition": "conversation",
  "condition_description": "Natural free conversation",
  "runs_combined": [1, 3, 5],
  "n_runs": 3,
  "gap_duration_s": 2.0,
  "concatenation_boundaries_s": [385.2, 770.8],
  "total_duration_s": 1155.4,
  "n_words_total": 2847,
  "bad_annotations": 2
}
```

---

## Using Combined Files for TRF Analysis

### Load Combined Data

```python
import mne
import numpy as np

# Load conversation condition
raw_conv = mne.io.read_raw_fif(
    'outputs/trf_combined/sub-01/sub-01_conversation_trf_raw.fif',
    preload=True
)

# Load nursery rhyme condition
raw_nursery = mne.io.read_raw_fif(
    'outputs/trf_combined/sub-01/sub-01_nursery_rhyme_trf_raw.fif',
    preload=True
)

# Check BAD annotations
print(f"Conversation BAD segments: {len(raw_conv.annotations)}")
print(f"Nursery rhyme BAD segments: {len(raw_nursery.annotations)}")
```

### Extract Predictors (excluding BAD segments)

```python
# Get predictor data
word_onsets_int = raw_conv.get_data(picks='MISC_word_onsets_interviewer')[0]
surprisal_int = raw_conv.get_data(picks='MISC_surprisal_interviewer')[0]
speaker = raw_conv.get_data(picks='MISC_speaker')[0]

# MNE automatically handles BAD annotations in epoching/TRF tools
# But for manual analysis, you can mask them:
mask = np.ones(len(word_onsets_int), dtype=bool)

for annot in raw_conv.annotations:
    if 'BAD' in annot['description']:
        start_idx = int(annot['onset'] * raw_conv.info['sfreq'])
        end_idx = start_idx + int(annot['duration'] * raw_conv.info['sfreq'])
        mask[start_idx:end_idx] = False

# Clean data
word_onsets_clean = word_onsets_int[mask]
```

### Compare Conditions with Eelbrain Boosting

```python
import eelbrain
import mne

# Load both conditions for one subject
raw_conv = mne.io.read_raw_fif('sub-01_conversation_trf_raw.fif', preload=True)
raw_nursery = mne.io.read_raw_fif('sub-01_nursery_rhyme_trf_raw.fif', preload=True)

# Convert to eelbrain (automatically respects BAD annotations)
conv_ndvar = eelbrain.load.fiff.mne_raw(raw_conv, exclude='bads')
nursery_ndvar = eelbrain.load.fiff.mne_raw(raw_nursery, exclude='bads')

# Extract predictors
conv_predictors = {
    'word_onsets': conv_ndvar['MISC_word_onsets_interviewer'],
    'surprisal': conv_ndvar['MISC_surprisal_interviewer'],
}

nursery_predictors = {
    'word_onsets': nursery_ndvar['MISC_word_onsets_interviewer'],
    'surprisal': nursery_ndvar['MISC_surprisal_interviewer'],
}

# Fit TRFs separately
trf_conv = eelbrain.boosting(
    conv_ndvar['meg'],
    list(conv_predictors.values()),
    tstart=-0.1,
    tstop=0.5,
)

trf_nursery = eelbrain.boosting(
    nursery_ndvar['meg'],
    list(nursery_predictors.values()),
    tstart=-0.1,
    tstop=0.5,
)

# Compare conditions
# Conversation should show stronger anticipatory activity (negative lags)
# Nursery rhyme should show predictable response patterns
```

---

## Expected Differences Between Conditions

### Conversation (Natural Turn-Taking)

**TRF Characteristics:**
- **Pre-word activity** (negative lags): Anticipatory signals before partner's words
- **Variable responses**: Different TRF weights across words (unpredictable)
- **Turn boundary effects**: Neural signatures of spontaneous turn transitions
- **Surprisal effects**: Stronger modulation by word predictability

**Regions of Interest:**
- **Left IFG**: Speech planning, turn prediction
- **Left STG**: Auditory prediction, comprehension
- **Motor cortex**: Anticipatory motor preparation

### Nursery Rhyme (Structured Repetition)

**TRF Characteristics:**
- **Reduced pre-word activity**: Less anticipation (participant knows what's coming)
- **Consistent responses**: Similar TRF weights across repetitions
- **Predictable timing**: Regular turn intervals
- **Reduced surprisal effects**: All words highly predictable

**Regions of Interest:**
- **Bilateral STG**: Auditory processing, repetition
- **Motor cortex**: Speech production during repetition
- **Cerebellum**: Timing, sequence learning

---

## Statistical Comparison

### Approach 1: Contrast TRFs Directly

```python
# Compute difference in TRF weights
trf_diff = trf_conv.h_scaled['word_onsets'] - trf_nursery.h_scaled['word_onsets']

# Test significance
result = eelbrain.testnd.ttest_1samp(
    trf_diff,
    tail=0,  # Two-tailed
    samples=10000,
)

# Plot significant regions
eelbrain.plot.brain.dspm(result.masked_parameter_map())
```

### Approach 2: Within-Subject Mixed Model

```python
# Stack conditions with labels
conditions = ['conversation'] * len(subjects) + ['nursery_rhyme'] * len(subjects)

# Extract TRF weights for each subject/condition
trf_weights = []  # Shape: (n_subjects * 2, n_channels, n_lags)

# Run mixed-effects model
# Condition as fixed effect, Subject as random effect
```

---

## Quality Checks

### 1. Verify Concatenation Boundaries

```python
import json

with open('outputs/trf_combined/sub-01/sub-01_conversation_metadata.json') as f:
    meta = json.load(f)

print(f"Runs combined: {meta['runs_combined']}")
print(f"Boundaries at: {meta['concatenation_boundaries_s']}")
print(f"Total duration: {meta['total_duration_s']:.1f}s")
```

### 2. Check BAD Annotations

```python
raw = mne.io.read_raw_fif('sub-01_conversation_trf_raw.fif')

for annot in raw.annotations:
    print(f"{annot['description']}: {annot['onset']:.2f}s ({annot['duration']:.2f}s)")
```

Expected output:
```
BAD_concatenation_boundary: 383.20s (2.00s)
BAD_concatenation_boundary: 768.80s (2.00s)
```

### 3. Visualize Combined Data

```python
# Plot with BAD regions shaded
raw.plot(duration=30, n_channels=20, scalings='auto')
# BAD regions will be highlighted in red
```

---

## Troubleshooting

### Issue: Missing Runs

**Error:** `✗ Missing run 3: outputs/trf/sub-01/run-03/sub-01_run-03_trf_raw.fif`

**Solution:** Run the full pipeline for missing runs:
```bash
python scripts/transcribe_dual_speaker.py --subject sub-01 --runs 3
python scripts/whisper_to_mfa.py --subject sub-01 --runs 3
python scripts/run_mfa_alignment.py --subject sub-01 --runs 3
python scripts/create_trf_fif.py --subject sub-01 --runs 3
```

### Issue: Channel Mismatch

**Error:** `Cannot concatenate - channel names don't match`

**Solution:** All runs must have identical channel configuration. Re-run `create_trf_fif.py` with `--overwrite` to ensure consistency.

### Issue: Memory Error

**Error:** `MemoryError: Unable to allocate array`

**Solution:** Process subjects individually or reduce the number of MEG channels:
```python
# Pick only gradiometers
raw.pick_types(meg='grad')
```

---

## Batch Processing

### Process All Subjects

```bash
# Create combined files for all subjects
python scripts/combine_runs.py --all

# This creates:
# - 28 subjects × 2 conditions = 56 combined files
# - Processing time: ~2-3 minutes per subject
```

### Parallel Processing (if needed)

```bash
# Process subjects in parallel using GNU parallel
parallel -j 4 python scripts/combine_runs.py --subject {} ::: sub-01 sub-02 sub-03 sub-04
```

---

## Next Steps

1. **Combine runs** for all subjects:
   ```bash
   python scripts/combine_runs.py --all
   ```

2. **Verify outputs**:
   ```bash
   ls -lh outputs/trf_combined/*/
   ```

3. **Run TRF analysis** comparing conditions:
   - Natural conversation: Anticipatory turn-taking signals
   - Nursery rhyme: Structured repetition baseline

4. **Test hypotheses**:
   - H1: Conversation shows stronger pre-word activity in IFG
   - H2: Conversation shows larger surprisal effects
   - H3: Nursery rhyme shows more consistent timing

---

**Ready to compare natural vs. structured turn-taking! 🚀**
