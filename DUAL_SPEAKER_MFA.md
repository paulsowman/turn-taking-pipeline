# Dual-Speaker MFA Workflow

**Date:** 2025-11-13
**Status:** ✅ Implemented

---

## Overview

This pipeline now supports speaker-specific word timing using Montreal Forced Aligner (MFA) on separate audio channels. This provides precise (±10-20ms) word-level timing for both interviewer and participant speech.

### What's New

- **13 TRF predictor channels** (was 10)
- **Speaker-specific word predictors**: onsets, surprisal, duration for each speaker
- **Dual-speaker MFA alignment**: Separate forced alignment for interviewer and participant

---

## Complete Workflow

### Step 1: Transcribe Both Speakers Separately

```bash
# Activate your virtual environment
source venv/bin/activate

# Transcribe both audio channels separately
python scripts/transcribe_dual_speaker.py --subject sub-01 --run 1
```

**Output:**
- `outputs/features/sub-01/run-01/transcript_interviewer.csv`
- `outputs/features/sub-01/run-01/transcript_participant.csv`

**Time:** ~20-30 seconds per subject (Whisper runs twice)

### Step 2: Convert to MFA Format

```bash
# Activate MFA conda environment
conda activate mfa

# Convert both transcripts to MFA format
python scripts/whisper_to_mfa.py --subject sub-01 --run 1
```

**Output:**
- `outputs/mfa/sub-01/run-01/interviewer/` (audio + text)
- `outputs/mfa/sub-01/run-01/participant/` (audio + text)

**Time:** <1 second

### Step 3: Run MFA Alignment

```bash
# Still in MFA conda environment
python scripts/run_mfa_alignment.py --subject sub-01 --run 1
```

**Output:**
- `outputs/features/sub-01/run-01/transcript_mfa_interviewer.csv`
- `outputs/features/sub-01/run-01/transcript_mfa_participant.csv`
- `outputs/features/sub-01/run-01/phones_mfa_interviewer.csv`
- `outputs/features/sub-01/run-01/phones_mfa_participant.csv`

**Time:** 1-2 minutes per subject (MFA runs twice)

### Step 4: Create TRF FIF File

```bash
# Exit MFA conda environment, activate venv
conda deactivate
source venv/bin/activate

# Create TRF FIF with 13 predictor channels
python scripts/create_trf_fif.py --subject sub-01 --run 1
```

**Output:**
- `outputs/trf/sub-01/run-01/sub-01_run-01_trf_raw.fif`
- `outputs/trf/sub-01/run-01/sub-01_run-01_predictor_metadata.json`

**Time:** 2-5 minutes (includes GPT-2 surprisal computation)

**Skip surprisal for faster testing:**
```bash
python scripts/create_trf_fif.py --subject sub-01 --run 1 --no-surprisal
```

---

## Predictor Channels (13 Total)

### Audio Envelopes (4 channels)
1. **MISC_envelope_interviewer**: External interviewer audio envelope (0-1)
2. **MISC_envelope_participant**: External participant audio envelope (0-1)
3. **MISC_envelope_meg_mic7**: MEG-internal MISC 007 envelope (0-1)
4. **MISC_envelope_meg_mic8**: MEG-internal MISC 008 envelope (0-1)

### Fundamental Frequency (2 channels)
5. **MISC_f0_interviewer**: Interviewer F0 contour, normalized 0-1 (voiced only)
6. **MISC_f0_participant**: Participant F0 contour, normalized 0-1 (voiced only)

### Word Onsets (2 channels) - NEW!
7. **MISC_word_onsets_interviewer**: Delta functions at interviewer word onsets
8. **MISC_word_onsets_participant**: Delta functions at participant word onsets

### Surprisal (2 channels) - NEW!
9. **MISC_surprisal_interviewer**: GPT-2 surprisal for interviewer words (0-1)
10. **MISC_surprisal_participant**: GPT-2 surprisal for participant words (0-1)

### Duration (2 channels) - NEW!
11. **MISC_duration_interviewer**: Interviewer word durations (0-1)
12. **MISC_duration_participant**: Participant word durations (0-1)

### Speaker (1 channel)
13. **MISC_speaker**: Speaker state (0=silence, 1=interviewer, 2=participant, 3=overlap)

---

## Viewing the FIF File

```bash
# Using MNE command-line viewer
mne browse_raw outputs/trf/sub-01/run-01/sub-01_run-01_trf_raw.fif

# Or in Python
import mne
raw = mne.io.read_raw_fif('outputs/trf/sub-01/run-01/sub-01_run-01_trf_raw.fif')
raw.plot(scalings='auto')
```

**Tip:** Use MNE viewer's channel selection to compare:
- Interviewer vs participant envelopes
- External vs MEG-internal audio (for sync verification)
- Word onsets for each speaker

---

## Advantages of Dual-Speaker MFA

### Timing Accuracy
- **MFA**: ±10-20ms word onset precision (phonetic alignment)
- **Whisper**: ±50-200ms word onset precision (VAD-based)

### Speaker Separation
- **Before**: Combined word predictors (ambiguous speaker)
- **Now**: Separate predictors for each speaker (clear attribution)

### Turn-Taking Analysis
You can now model:
- **Speaker-specific TRFs**: Do interviewer and participant words elicit different neural responses?
- **Turn transitions**: Neural activity when switching speakers vs. continuing
- **Overlap effects**: Neural correlates of simultaneous speech

---

## Batch Processing

```bash
# Transcribe all subjects
for sub in sub-01 sub-02 sub-03; do
    python scripts/transcribe_dual_speaker.py --subject $sub --run 1
done

# Convert to MFA
conda activate mfa
for sub in sub-01 sub-02 sub-03; do
    python scripts/whisper_to_mfa.py --subject $sub --run 1
done

# Run MFA alignment
for sub in sub-01 sub-02 sub-03; do
    python scripts/run_mfa_alignment.py --subject $sub --run 1
done

# Create TRF FIFs
conda deactivate
source venv/bin/activate
for sub in sub-01 sub-02 sub-03; do
    python scripts/create_trf_fif.py --subject $sub --run 1
done
```

---

## Files Modified

1. **scripts/transcribe_dual_speaker.py** (NEW)
   - Transcribes both speakers separately using Whisper

2. **scripts/whisper_to_mfa.py** (UPDATED)
   - Now processes both speaker transcripts
   - Creates separate MFA directories for each speaker

3. **scripts/run_mfa_alignment.py** (UPDATED)
   - Runs MFA on both speakers' audio separately
   - Saves speaker-specific MFA outputs

4. **scripts/create_trf_fif.py** (UPDATED)
   - Loads both speaker transcripts
   - Creates speaker-specific predictors (onsets, surprisal, duration)
   - Generates 13 predictor channels (was 10)

---

## Troubleshooting

### Issue: Missing Transcript Files

**Error:** `✗ Interviewer transcript not found: transcript_interviewer.csv`

**Solution:** Run Step 1 first:
```bash
python scripts/transcribe_dual_speaker.py --subject sub-01 --run 1
```

### Issue: MFA Timeout

**Error:** `Command 'mfa version' timed out after 5 seconds`

**Solution:** MFA v3.x on macOS is slow to start. Already fixed in code (timeout=30s).

### Issue: Wrong Channel for Participant

**Error:** Participant audio is silent or incorrect

**Solution:** The code auto-detects channels:
- Separate file (subject_mic) → channel 0
- Same file (console_mic) → channel 1

Check `AUDIO_CHANNEL_STRUCTURE.md` for your setup.

---

## Example TRF Analysis

```python
import mne
import numpy as np

# Load TRF FIF file
raw = mne.io.read_raw_fif('outputs/trf/sub-01/run-01/sub-01_run-01_trf_raw.fif')

# Extract speaker-specific predictors
interviewer_onsets = raw.get_data(picks='MISC_word_onsets_interviewer')[0]
participant_onsets = raw.get_data(picks='MISC_word_onsets_participant')[0]

# Compare word rates
int_rate = np.sum(interviewer_onsets > 0) / (len(interviewer_onsets) / raw.info['sfreq'])
part_rate = np.sum(participant_onsets > 0) / (len(participant_onsets) / raw.info['sfreq'])

print(f"Interviewer: {int_rate:.2f} words/s")
print(f"Participant: {part_rate:.2f} words/s")

# Load MEG data for TRF modeling
meg_picks = mne.pick_types(raw.info, meg=True)
meg_data = raw.get_data(picks=meg_picks)

# Now run TRF with eelbrain boosting...
```

---

## Next Steps

### Validate Dual-Speaker Setup
1. Run on sub-01 run-01 as a test case
2. Check metadata file for word counts per speaker
3. Verify predictor quality in MNE viewer

### Run on Full Dataset
```bash
# Process all subjects and runs
python scripts/transcribe_dual_speaker.py --all
python scripts/whisper_to_mfa.py --all
python scripts/run_mfa_alignment.py --all
python scripts/create_trf_fif.py --all
```

### TRF Modeling
- Compare interviewer vs. participant TRF responses
- Model turn-taking transitions with speaker predictor
- Test linguistic predictors (surprisal) for each speaker

---

**Ready for speaker-specific TRF analysis! 🚀**
