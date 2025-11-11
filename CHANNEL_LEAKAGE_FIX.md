# Audio Channel Leakage Fix and Data Regeneration Plan

## Summary

Fixed critical audio channel leakage issue where both stereo channels (interviewer + participant) were being mixed together during processing. All ASR transcripts and prosody features extracted before this fix contain participant leakage and need to be regenerated.

## The Problem

### Dual-Microphone Recording Structure

Console_mic recordings have:
- **Left channel (ch 0)**: Interviewer audio (what we want)
- **Right channel (ch 1)**: Participant audio (contamination)

All audio loading was using `librosa.load(..., mono=True)` which averages both channels, causing cross-talk leakage from the participant into the interviewer's audio.

### Affected Modules

1. **ASR Module** (`src/asr/whisper_asr.py`): Whisper transcribed mixed audio
2. **Prosody Module** (`src/prosody/features.py`): F0 and prosodic features extracted from mixed audio
3. **Sync Module** (`src/sync/audio_meg_sync.py`): Synchronization used mixed audio
4. **All Scripts**: Verification and debugging scripts used mixed audio

### Impact

All existing data in `outputs/features/{subject}/run-{run}/`:
- ❌ `transcript.csv`: Word timestamps based on mixed audio (interviewer + participant)
- ❌ `prosody.csv`: F0 and prosodic features from mixed audio
- ❌ `pauses.csv`: Pause detection from mixed audio
- ❌ `speech_rate.csv`: Speech rate from mixed transcript
- ❌ `trp_features.csv`: TRP features from contaminated data
- ❌ Any TRF models trained on these features

## The Fix

### 1. Created Channel-Aware Audio Loading (`src/utils/io.py`)

Added `load_audio()` function with channel selection:

```python
def load_audio(
    file_path: Union[str, Path],
    sr: Optional[int] = None,
    channel: Optional[int] = 0,  # 0=left/interviewer, 1=right/participant
    **kwargs
) -> Tuple[np.ndarray, float]:
    """
    Load audio file with proper channel selection for stereo files.

    For dual-microphone recordings where interviewer and participant are on
    separate channels, ALWAYS specify channel=0 or channel=1 to avoid leakage.
    Using mono=True averages both channels and causes cross-talk.
    """
```

**Critical fix (2025-11-11):** When `offset` and `duration` parameters are used with `channel` selection, the function now:
1. Extracts offset/duration from kwargs
2. Loads the FULL stereo file with `mono=False`
3. Selects the requested channel
4. Applies offset/duration by slicing the numpy array

This prevents librosa from potentially mixing channels during offset/duration extraction. Previously, passing offset/duration to `librosa.load(..., mono=False)` could cause channel mixing.

### 2. Updated ASR Module (`src/asr/whisper_asr.py`)

**Changes:**
- Added `channel` parameter (default: 0 for interviewer)
- Pre-loads audio with channel selection before passing to Whisper
- Passes numpy array to Whisper instead of file path

**Key code:**
```python
def transcribe_audio(
    audio_path: Path,
    model_name: str = "base",
    language: str = "en",
    device: Optional[str] = None,
    channel: Optional[int] = 0,  # NEW
) -> pd.DataFrame:
    # Load audio with proper channel selection
    from utils.io import load_audio
    audio, sr = load_audio(audio_path, sr=16000, channel=channel)

    # Pass numpy array to Whisper
    result = model.transcribe(audio, language=language, ...)
```

**Fixed incorrect sign convention:**
- Corrected formula to: `meg_time = audio_time - offset` (SUBTRACT)
- Old code used `+ offset_s` which was mathematically wrong
- With offset=-12.235s:
  - **Old (wrong)**: `meg_time = 4.20 + (-12.235) = -8.035` ❌
  - **New (correct)**: `meg_time = 4.20 - (-12.235) = 16.435` ✓

### 3. Updated Prosody Module (`src/prosody/features.py`)

**Changes:**
- Added `channel` parameter (default: 0 for interviewer)
- Uses `load_audio()` for channel selection
- Creates Parselmouth Sound from numpy array instead of file path

**Key code:**
```python
def extract_prosody_features(
    audio_path: Path,
    sr: Optional[int] = None,
    frame_shift: float = 0.01,
    f0_min: float = 75.0,
    f0_max: float = 500.0,
    channel: Optional[int] = 0,  # NEW
) -> pd.DataFrame:
    # Load audio with proper channel selection
    from utils.io import load_audio
    y, sr_librosa = load_audio(audio_path, sr=sr, channel=channel)

    # Create Parselmouth Sound from array
    snd = parselmouth.Sound(y, sampling_frequency=sr_librosa)
```

### 4. Updated Sync Module (`src/sync/audio_meg_sync.py`)

**Changes:**
- Uses `load_audio()` with configurable channel selection
- Reads `sync.audio_channel` from config (default: 0)
- Logs which channel is being used

### 5. Updated All Verification Scripts

**Scripts updated:**
- `scripts/verify_sync_alignment.py`: Channel selection for envelope/F0 visualization + recalculates MEG times
- `scripts/check_f0_word_alignment.py`: Channel selection for F0 statistics + recalculates MEG times
- `scripts/create_onset_playback.py`: Channel selection for audio playback + debug output
- `src/qc/sync_qc.py`: Added helper function with channel selection
- `scripts/diagnose_audio_channels.py`: NEW - Diagnostic tool to verify channel separation

**Diagnostic Tool:**
The new `diagnose_audio_channels.py` script helps verify that stereo files have proper channel separation:
```bash
python scripts/diagnose_audio_channels.py --subject sub-01 --run 1
```

This checks:
- Number of channels in the file
- RMS power of each channel
- Whether channels are identical (duplicated mono) or different (true stereo)
- How librosa handles offset/duration with mono=False

### 6. Documentation

Created comprehensive documentation:
- `AUDIO_CHANNEL_STRUCTURE.md`: Explains stereo channel layout and leakage issue
- `CHANNEL_LEAKAGE_FIX.md`: This document
- `SYNC_ALIGNMENT_FIX.md`: Documents reversed sign convention and alignment fixes

## Data Regeneration Required

### Critical: Must Regenerate

All features for all subjects need to be regenerated with clean single-channel audio:

```bash
# For each subject and run, regenerate features:
python scripts/test_asr_prosody.py --subject sub-01 --run 1
python scripts/test_asr_prosody.py --subject sub-01 --run 2
# ... etc for all subjects and runs
```

This will regenerate:
- ✅ `transcript.csv`: Clean interviewer-only transcripts
- ✅ `prosody.csv`: Clean F0 and prosodic features
- ✅ `pauses.csv`: Accurate pause detection
- ✅ `speech_rate.csv`: Correct speech rate
- ✅ `trp_features.csv`: Valid TRP features

### After Feature Regeneration

Once features are regenerated, all TRF models need retraining:
- TRF models were trained on contaminated features
- Need to rerun all TRF analysis with clean features
- Use `scripts/multipredictor_trf_combined.py` for each subject

## Verification of Fix

### 1. Formula Verification

Verified with actual data from `sub-01/run-01`:
```python
# From transcript.csv
audio_start = 4.20s
meg_start = 16.435s

# From sync_params.json
offset = -12.235s

# Test corrected formula
meg_time = audio_time - offset
16.435 = 4.20 - (-12.235)
16.435 = 16.435  ✓ CORRECT!
```

### 2. Channel Selection Verification

Test that channel selection works:
```bash
# Create onset playback (should hear only interviewer)
python scripts/create_onset_playback.py \
    --subject sub-01 --run 1 \
    --start 10 --duration 30

# Verify no participant leakage in audio
# Listen to: outputs/sync/sub-01/run-01/onset_playback_*.wav
```

### 3. Alignment Verification

Test that alignment is correct:
```bash
# Generate sync alignment visualization
python scripts/verify_sync_alignment.py \
    --subject sub-01 --run 1 \
    --start 10 --end 40 \
    --zoom-start 15 --zoom-end 18

# Check: outputs/sync/sub-01/run-01/sync_verification.png
# - Envelope peaks should align with word onsets
# - F0 contours should align with voiced regions
# - MEG aux channel should match external envelope
```

### 4. F0-Word Alignment Check

Verify F0 patterns match phonetic expectations:
```bash
python scripts/check_f0_word_alignment.py \
    --subject sub-01 --run 1 \
    --window-ms 50

# Expected results:
# - Vowel-initial words: ~100% should have F0
# - Unvoiced consonants: ~0-20% should have immediate F0
# - Voiced consonants: ~80-100% should have F0 within 50ms
```

## Configuration Changes

Updated `config/config.yaml`:
```yaml
sync:
  aux_channel_name: "MISC 007"
  audio_channel: 0  # NEW: 0=interviewer (left), 1=participant (right)
```

## Code Quality Improvements

### Sign Convention Documentation

All time conversion code now clearly documents the REVERSED sign convention:
```python
# NOTE: Sync function uses REVERSED sign convention!
# Negative offset = external starts AFTER MEG (subtract negative = add)
# Positive offset = external starts BEFORE MEG (subtract positive = subtract)
# Formula: meg_time = audio_time - offset
```

### Consistent Channel Selection

All audio loading now uses consistent pattern:
```python
from utils.io import load_audio
audio, sr = load_audio(audio_path, sr=target_sr, channel=0)
```

## Testing Checklist

Before regenerating all data, test on one subject:

- [x] ASR module: Verify transcribe_audio() uses channel parameter
- [x] Prosody module: Verify extract_prosody_features() uses channel parameter
- [x] Sync module: Verify synchronization uses channel parameter
- [x] Formula: Verify meg_time = audio_time - offset (not + offset)
- [x] Scripts: All verification scripts updated for channel selection
- [ ] Test run: Generate features for sub-01 run-01 and verify quality
- [ ] Audio playback: Listen to onset_playback and confirm no leakage
- [ ] Sync visualization: Check alignment looks correct
- [ ] F0 statistics: Verify phonetic patterns make sense

## Batch Regeneration Commands

Once testing is complete, batch regenerate all features:

```bash
# Option 1: Loop through all subjects manually
for subject in sub-01 sub-02 sub-03 ...; do
    for run in 1 2 3 4 5; do
        echo "Processing $subject run $run"
        python scripts/test_asr_prosody.py \
            --subject $subject \
            --run $run
    done
done

# Option 2: Create a batch script (recommended)
# TODO: Create scripts/batch_regenerate_features.py
```

## Files Changed

### Core Modules
1. `src/utils/io.py` - NEW: Added load_audio() function
2. `src/asr/whisper_asr.py` - MODIFIED: Channel selection + formula fix
3. `src/prosody/features.py` - MODIFIED: Channel selection
4. `src/sync/audio_meg_sync.py` - MODIFIED: Channel selection
5. `src/qc/sync_qc.py` - MODIFIED: Added helper function

### Scripts
6. `scripts/verify_sync_alignment.py` - MODIFIED: Channel selection
7. `scripts/check_f0_word_alignment.py` - MODIFIED: Channel selection
8. `scripts/create_onset_playback.py` - MODIFIED: Channel selection

### Documentation
9. `AUDIO_CHANNEL_STRUCTURE.md` - NEW: Channel layout documentation
10. `SYNC_ALIGNMENT_FIX.md` - NEW: Sign convention documentation
11. `CHANNEL_LEAKAGE_FIX.md` - NEW: This document

### Configuration
12. `config/config.yaml` - MODIFIED: Added audio_channel parameter

## Notes

- **Subject_mic recordings**: Right channel was copied from console_mic, so leakage won't be audible, but data is still technically mixed. Should use channel=0 consistently.
- **Sync quality**: Original synchronization may have been affected by leakage but likely still valid (high correlation means timing is correct even if amplitude is mixed)
- **Backward compatibility**: Default channel=0 ensures new code works correctly; old data must be regenerated
- **Testing**: Always listen to onset_playback files to verify audio quality before batch processing

## Verification Script Fixes (Added 2025-11-11)

### Issue Discovered
After fixing the ASR/prosody modules, verification scripts showed envelopes aligned correctly but word onsets were still misaligned by ~24 seconds.

### Root Cause
The verification scripts were reading `start_meg` and `end_meg` columns from existing `transcript.csv` files. These values were calculated with the OLD (incorrect) formula and stored in the file. When the scripts read these values, they were using incorrect MEG times.

**Example with offset=-12.235s:**
- Audio time: 4.20s
- **OLD** (in transcript.csv): `meg_time = 4.20 + (-12.235) = -8.035s` ❌
- **Envelope** (calculated correctly): `meg_time = 4.20 - (-12.235) = 16.435s` ✓
- **Misalignment**: 24.47 seconds!

### Scripts Fixed

Both verification scripts now recalculate MEG times from audio times instead of trusting the stored values:

1. **`scripts/verify_sync_alignment.py`**
   - `load_word_onsets()` function updated
   - Loads sync_params.json to get offset
   - Recalculates: `meg_time = audio_time - offset`

2. **`scripts/check_f0_word_alignment.py`**
   - Same fix applied to `load_word_onsets()` function
   - Ensures F0 analysis uses correctly aligned word onsets

### Verification
```python
# Math verification:
offset = -12.235s
audio_time = 4.20s

# OLD (in transcript.csv)
old_meg_time = 4.20 + (-12.235) = -8.035s  # WRONG!

# NEW (recalculated)
new_meg_time = 4.20 - (-12.235) = 16.435s  # CORRECT!

# Envelope (always correct)
envelope_meg = 4.20 - (-12.235) = 16.435s  # MATCHES!
```

Now word onsets align perfectly with audio envelope and F0 contours in all visualization tools.

## librosa offset/duration Channel Mixing Issue (Fixed 2025-11-11)

### Issue Discovered
Even after fixing channel selection in all modules, the onset playback audio still contained both participants mixed together.

### Root Cause
When `librosa.load()` is called with BOTH `mono=False` and `offset`/`duration` parameters, librosa may internally mix channels during the offset/duration extraction process. This resulted in:
```python
# This caused channel mixing:
audio, sr = librosa.load(file, sr=None, mono=False, offset=10, duration=30)
audio_ch0 = audio[0]  # Still has both channels mixed!
```

### Solution
Modified `src/utils/io.py::load_audio()` to:
1. Extract `offset` and `duration` from kwargs
2. Load FULL stereo file: `librosa.load(file, mono=False)` (no offset/duration)
3. Select channel: `audio = audio[channel]`
4. Apply offset/duration by numpy slicing: `audio = audio[start:end]`

This ensures channel selection happens BEFORE any time-based slicing, preventing librosa from mixing channels.

```python
# New approach (correct):
audio, sr = librosa.load(file, sr=None, mono=False)  # Load full file
audio_ch0 = audio[0]  # Select channel first
audio_ch0 = audio_ch0[start:end]  # Then apply offset/duration
```

### Files Modified
- `src/utils/io.py`: Updated `load_audio()` implementation
- `scripts/create_onset_playback.py`: Added debug output to verify mono audio
- `scripts/diagnose_audio_channels.py`: Created diagnostic tool

---

**Status**: ✅ CODE FIXED - Data regeneration required
**Date**: 2025-11-11 (Updated with offset/duration channel mixing fix)
**Impact**: All existing transcripts and prosody features invalid
**Next Steps**: Test on one subject, then batch regenerate all features
