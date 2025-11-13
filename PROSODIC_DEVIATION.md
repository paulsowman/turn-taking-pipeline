# Prosodic Deviation Predictors for TRF Analysis

**Date:** 2025-11-13
**Status:** ✅ Implemented

---

## Overview

The pipeline now includes **prosodic deviation predictors** that complement lexical surprisal with measures of prosodic unexpectedness. These enable TRF analysis to separate lexical predictability (what words are spoken) from prosodic predictability (how they are spoken).

### What's New

- **19 TRF predictor channels** (was 13)
- **6 new prosodic deviation predictors** (3 per speaker):
  - F0 deviation (pitch unexpectedness)
  - Duration deviation (timing unexpectedness)
  - Pause before word (inter-word intervals)

---

## Motivation

### Lexical vs. Prosodic Surprisal

**Lexical surprisal** measures word predictability from context:
- "What did you have for breakfast?" → "cereal" (low surprisal)
- "What did you have for breakfast?" → "elephant" (high surprisal)

**Prosodic surprisal** measures how a word is spoken:
- Normal pitch → expected (low deviation)
- Unusually high pitch → unexpected (high deviation)
- Very short duration → unexpected (high deviation)
- Long pause before word → unexpected (high pause value)

### Research Questions

1. **Separate neural signatures**: Do lexical and prosodic surprisal have different TRF latencies or brain regions?
2. **Interaction effects**: Does prosodic deviation modulate lexical surprisal effects?
3. **Turn-taking cues**: Do prosodic deviations predict upcoming turn transitions?

---

## Complete Channel List (19 Total)

### INTERVIEWER (9 channels)

#### Audio Features
1. **MISC_envelope_interviewer**: Audio envelope (external audio)
2. **MISC_envelope_meg_mic7**: MEG MISC 007 envelope (for sync verification)

#### Pitch Features
3. **MISC_f0_interviewer**: F0 contour (speaker-masked, normalized 0-1)
4. **MISC_f0_deviation_interviewer**: Z-scored F0 deviations from speaker mean

#### Word-Level Features
5. **MISC_word_onsets_interviewer**: Delta functions at word onsets
6. **MISC_surprisal_interviewer**: Lexical surprisal (GPT-2, conversation-aware)
7. **MISC_duration_interviewer**: Word durations (normalized 0-1)
8. **MISC_duration_deviation_interviewer**: Z-scored duration deviations
9. **MISC_pause_interviewer**: Time since last word (normalized 0-1)

### PARTICIPANT (9 channels)

Same as interviewer, but for participant speech.

### SHARED (1 channel)

19. **MISC_speaker**: Categorical (0=silence, 1=interviewer, 2=participant, 3=overlap)

---

## Prosodic Deviation Predictors Explained

### 1. F0 Deviation (Pitch Unexpectedness)

**Computation:**
```python
# For each speaker:
# 1. Compute mean F0 across all voiced frames
f0_mean = mean(f0_voiced)
f0_std = std(f0_voiced)

# 2. For each word, get mean F0 during that word
word_f0 = mean(f0[word_start:word_end])

# 3. Compute z-score
f0_deviation = (word_f0 - f0_mean) / f0_std
```

**Example values:**
- `+2.0`: Unusually high pitch (2 standard deviations above speaker mean)
- `0.0`: Typical pitch for this speaker
- `-1.5`: Lower pitch than usual

**Expected effects:**
- **High F0 deviation** at turn-final words: Prosodic cues for turn transition
- **Low F0 deviation** in nursery rhyme condition: Predictable prosody
- **Neural response**: May precede lexical surprisal in auditory cortex

### 2. Duration Deviation (Timing Unexpectedness)

**Computation:**
```python
# For each speaker:
# 1. Compute mean word duration
dur_mean = mean(word_durations)
dur_std = std(word_durations)

# 2. For each word, compute z-score
dur_deviation = (word_duration - dur_mean) / dur_std
```

**Example values:**
- `+2.5`: Unusually long word (drawn out)
- `0.0`: Typical duration
- `-1.2`: Shorter than usual (rushed)

**Expected effects:**
- **High duration deviation** before turn transitions: Pre-final lengthening
- **Negative duration deviation** in rapid speech: Reduced articulation
- **Neural response**: Motor cortex preparation for turn-taking

### 3. Pause Before Word (Inter-Word Interval)

**Computation:**
```python
# For each speaker:
# Compute time since previous word
pause_values[0] = 0  # First word has no previous
pause_values[i] = word_time[i] - word_time[i-1]  # i > 0

# Normalize to 0-1 range
pause_norm = (pause - pause_min) / (pause_max - pause_min)
```

**Example values:**
- `1.0`: Maximum pause for this speaker (long silence before word)
- `0.5`: Moderate pause
- `0.0`: Minimum pause (words close together)

**Expected effects:**
- **High pause values** at turn boundaries: Natural turn gaps
- **Low pause values** within turns: Fluent speech
- **Neural response**: Anticipatory activity during silence before partner's speech

---

## Usage

### Generate TRF FIF Files with Prosodic Predictors

```bash
# Single subject/run
python scripts/create_trf_fif.py --subject sub-01 --run 1

# All subjects, all runs
python scripts/create_trf_fif.py --all

# Skip surprisal for faster testing
python scripts/create_trf_fif.py --subject sub-01 --run 1 --no-surprisal
```

**Output:**
- `outputs/trf/sub-01/run-01/sub-01_run-01_trf_raw.fif` (19 MISC channels + MEG)
- `outputs/trf/sub-01/run-01/sub-01_run-01_predictor_metadata.json`

### Metadata Example

```json
{
  "f0_deviation_interviewer": {
    "n_values": 423,
    "mean_z": 0.02,
    "std_z": 0.98,
    "note": "Z-scored F0 deviations from speaker mean (prosodic unexpectedness)"
  },
  "duration_deviation_interviewer": {
    "n_values": 423,
    "mean_z": -0.01,
    "std_z": 1.01,
    "note": "Z-scored duration deviations from speaker mean"
  },
  "pause_interviewer": {
    "n_values": 423,
    "normalized": true,
    "note": "Time since last interviewer word (inter-word interval), normalized to 0-1"
  }
}
```

---

## TRF Analysis Examples

### Example 1: Compare Lexical vs. Prosodic Surprisal

```python
import eelbrain
import mne

# Load TRF FIF file
raw = mne.io.read_raw_fif('sub-01_run-01_trf_raw.fif', preload=True)

# Convert to eelbrain
meg_ndvar = eelbrain.load.fiff.mne_raw(raw)

# Extract predictors
predictors = {
    'lexical_surprisal': meg_ndvar['MISC_surprisal_interviewer'],
    'f0_deviation': meg_ndvar['MISC_f0_deviation_interviewer'],
    'duration_deviation': meg_ndvar['MISC_duration_deviation_interviewer'],
    'pause': meg_ndvar['MISC_pause_interviewer'],
}

# Fit TRF with all predictors
trf = eelbrain.boosting(
    meg_ndvar['meg'],
    list(predictors.values()),
    tstart=-0.1,
    tstop=0.6,
    basis=0.050,
)

# Compare TRF latencies
print("Lexical surprisal peak:", trf.h_scaled['lexical_surprisal'].argmax())
print("F0 deviation peak:", trf.h_scaled['f0_deviation'].argmax())
print("Duration deviation peak:", trf.h_scaled['duration_deviation'].argmax())
```

**Expected results:**
- **Lexical surprisal**: Peak ~200-300ms (semantic processing)
- **F0 deviation**: Peak ~100-150ms (auditory processing)
- **Duration deviation**: Peak ~150-250ms (timing prediction)

### Example 2: Turn-Taking Prediction

```python
# Focus on turn-final words
turns = pd.read_csv('outputs/features/sub-01/run-01/turns.csv')

# Extract prosodic features at turn-final words
turn_final_f0_dev = []
turn_final_dur_dev = []
turn_final_pause = []

for _, turn in turns.iterrows():
    if turn['speaker'] == 'interviewer':
        turn_end_time = turn['end']

        # Get predictor values at turn-final word
        f0_dev = raw['MISC_f0_deviation_interviewer'][0, closest_idx(turn_end_time)]
        dur_dev = raw['MISC_duration_deviation_interviewer'][0, closest_idx(turn_end_time)]
        pause = raw['MISC_pause_interviewer'][0, closest_idx(turn_end_time)]

        turn_final_f0_dev.append(f0_dev)
        turn_final_dur_dev.append(dur_dev)
        turn_final_pause.append(pause)

# Test: Are turn-final words prosodically marked?
print(f"Turn-final F0 deviation: {np.mean(turn_final_f0_dev):.2f}")
print(f"Turn-final duration deviation: {np.mean(turn_final_dur_dev):.2f}")
print(f"Turn-final pause: {np.mean(turn_final_pause):.2f}")
```

**Hypothesis:**
- Turn-final words should have **higher F0 deviation** (intonation cues)
- Turn-final words should have **positive duration deviation** (lengthening)
- Next word should have **higher pause** (gap before partner's speech)

### Example 3: Conversation vs. Nursery Rhyme

```python
# Load both conditions
raw_conv = mne.io.read_raw_fif('sub-01_conversation_trf_raw.fif')
raw_nursery = mne.io.read_raw_fif('sub-01_nursery_rhyme_trf_raw.fif')

# Compare prosodic variability
conv_f0_std = raw_conv['MISC_f0_deviation_interviewer'][0].std()
nursery_f0_std = raw_nursery['MISC_f0_deviation_interviewer'][0].std()

print(f"Conversation F0 variability: {conv_f0_std:.2f}")
print(f"Nursery rhyme F0 variability: {nursery_f0_std:.2f}")
```

**Expected result:**
- **Conversation**: Higher prosodic variability (spontaneous)
- **Nursery rhyme**: Lower prosodic variability (repetitive, predictable)

---

## Expected Neural Effects

### Lexical Surprisal (Existing)
- **Regions**: Left STG, IFG (semantic processing)
- **Latency**: 200-400ms
- **Effect**: Higher surprisal → Larger N400-like response

### F0 Deviation (NEW)
- **Regions**: Bilateral STG (auditory cortex)
- **Latency**: 100-200ms (earlier than lexical)
- **Effect**: Higher deviation → Larger auditory response

### Duration Deviation (NEW)
- **Regions**: Motor cortex, cerebellum (timing prediction)
- **Latency**: 150-300ms
- **Effect**: Unexpected durations → Prediction error signals

### Pause Before Word (NEW)
- **Regions**: IFG, motor cortex (anticipatory preparation)
- **Latency**: Pre-word activity (negative lags in TRF)
- **Effect**: Long pauses → Increased anticipatory activity

---

## Validation Checks

### Check 1: F0 Deviation Distribution

```python
import matplotlib.pyplot as plt

f0_dev = raw['MISC_f0_deviation_interviewer'][0]
f0_dev_nonzero = f0_dev[f0_dev != 0]

plt.hist(f0_dev_nonzero, bins=50)
plt.xlabel('F0 Deviation (z-scores)')
plt.ylabel('Count')
plt.title('Distribution should be centered near 0 with std ≈ 1')
plt.show()

print(f"Mean: {f0_dev_nonzero.mean():.3f} (should be ≈ 0)")
print(f"Std: {f0_dev_nonzero.std():.3f} (should be ≈ 1)")
```

### Check 2: Pause Values

```python
pause = raw['MISC_pause_interviewer'][0]
pause_nonzero = pause[pause != 0]

plt.hist(pause_nonzero, bins=50)
plt.xlabel('Pause (normalized)')
plt.ylabel('Count')
plt.title('Pauses should span 0-1 range')
plt.show()

print(f"Range: [{pause_nonzero.min():.3f}, {pause_nonzero.max():.3f}]")
print(f"Median: {np.median(pause_nonzero):.3f}")
```

### Check 3: Correlation Between Predictors

```python
# Check if prosodic and lexical surprisal are independent
import scipy.stats

surprisal = raw['MISC_surprisal_interviewer'][0]
f0_dev = raw['MISC_f0_deviation_interviewer'][0]

# Only compare at word onsets
onsets = raw['MISC_word_onsets_interviewer'][0] > 0
surp_at_words = surprisal[onsets]
f0_at_words = f0_dev[onsets]

r, p = scipy.stats.pearsonr(surp_at_words, f0_at_words)
print(f"Correlation between lexical surprisal and F0 deviation: r={r:.3f}, p={p:.4f}")
print("If r ≈ 0: Predictors are independent (good!)")
print("If r > 0.3: Predictors may be confounded")
```

---

## Troubleshooting

### Issue: F0 Deviation All Zeros

**Cause:** No voiced words detected

**Solution:**
```python
# Check metadata
with open('sub-01_run-01_predictor_metadata.json') as f:
    meta = json.load(f)
print(f"Voiced F0: {meta['predictors']['f0_interviewer']['pct_voiced']:.1f}%")
```

If pct_voiced is very low (<5%), check:
- Audio channel selection (participant vs. interviewer)
- F0 extraction parameters (fmin, fmax)

### Issue: Pause Values Not Normalized

**Error:** `Pause values outside 0-1 range`

**Cause:** Only one word detected (can't normalize)

**Solution:** Check word detection:
```python
n_words = meta['n_words_interviewer']
print(f"Words detected: {n_words}")
# Need at least 2 words to compute pauses
```

---

## Next Steps

### Immediate

1. **Regenerate TRF files** with prosodic predictors:
   ```bash
   python scripts/create_trf_fif.py --all
   ```

2. **Validate on sub-01**:
   - Check metadata for sensible statistics
   - Visualize predictor distributions
   - Verify no NaN or Inf values

### Analysis

3. **Test hypotheses**:
   - H1: Lexical and prosodic surprisal have different TRF latencies
   - H2: Turn-final words show prosodic marking (F0 deviation, lengthening)
   - H3: Nursery rhyme condition has lower prosodic variability

4. **Model comparison**:
   - TRF with lexical predictors only
   - TRF with prosodic predictors only
   - TRF with both (full model)
   - Compare prediction accuracy (r-values)

---

## References

### Prosodic Predictability

- **Turnbull et al. (2017)**: "Prosodic effects on word recognition: Evidence from prosodic prominence prediction"
- **Breen et al. (2010)**: "Acoustic correlates of information structure"
- **Cole et al. (2019)**: "Prosody in context: A review"

### Neural Correlates

- **Pitt & Samuel (1990)**: Duration and temporal predictability in spoken word recognition
- **Bögels et al. (2015)**: Neural signatures of turn-taking in conversation
- **Rimmele et al. (2015)**: The role of temporal structure in auditory processing

### TRF Methodology

- **Crosse et al. (2016)**: "The multivariate temporal response function (mTRF)"
- **Broderick et al. (2018)**: "Electrophysiological correlates of semantic dissimilarity"

---

**Ready to test lexical vs. prosodic prediction effects! 🎵**
