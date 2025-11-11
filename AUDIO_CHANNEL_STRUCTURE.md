# Audio Channel Structure and Configuration

## Recording Setup

The dual-microphone recording system uses the following channel structure:

### Console Mic Files (`console_mic_B{X}.wav`)

**Stereo recording with:**
- **Left channel (channel 0)**: Interviewer microphone
- **Right channel (channel 1)**: Participant microphone

**IMPORTANT:** When loading these files, you MUST specify `channel=0` to get only the interviewer, or `channel=1` to get only the participant. **NEVER use `mono=True`** as this averages both channels and causes cross-talk/leakage.

### Subject Mic Files (`subject_mic_B{X}.wav`)

**Processed stereo with:**
- **Left channel (channel 0)**: Participant microphone (copied from console_mic right channel)
- **Right channel (channel 1)**: Empty/silence

These files were created by extracting the participant channel from console_mic files, so using `channel=0` gives clean participant audio.

## Correct Usage

### Loading Interviewer Audio

```python
from utils.io import load_audio

# Load interviewer channel from console_mic
audio, sr = load_audio('console_mic_B1.wav', channel=0)  # ✓ CORRECT
```

### Loading Participant Audio

You can use either file:

```python
# Option 1: From console_mic (right channel)
audio, sr = load_audio('console_mic_B1.wav', channel=1)

# Option 2: From subject_mic (left channel, cleaner)
audio, sr = load_audio('subject_mic_B1.wav', channel=0)  # ✓ RECOMMENDED
```

### WRONG: Mixing Channels

```python
# ❌ WRONG - Averages both interviewer and participant
audio, sr = librosa.load('console_mic_B1.wav', sr=None, mono=True)
```

## Pipeline Configuration

The pipeline configuration (`config/config.yaml`) should include:

```yaml
sync:
  audio_channel: 0  # Default to left channel (interviewer for console_mic)
  # Set to 1 for right channel (participant) if needed
```

## Updated Modules

The following modules have been updated to use channel-specific loading:

1. **`src/utils/io.py`**: New `load_audio()` function with channel selection
2. **`src/sync/audio_meg_sync.py`**: Uses `load_audio()` with configurable channel
3. **`scripts/verify_sync_alignment.py`**: Loads channel 0 for interviewer
4. **`scripts/check_f0_word_alignment.py`**: Loads channel 0 for interviewer
5. **`scripts/create_onset_playback.py`**: Loads channel 0 for interviewer

## Channel Selection Guide

| Use Case | File | Channel | Configuration |
|----------|------|---------|---------------|
| Interview analysis (default) | `console_mic_BX.wav` | 0 (left) | `audio_channel: 0` |
| Participant analysis | `subject_mic_BX.wav` | 0 (left) | - |
| Participant from console | `console_mic_BX.wav` | 1 (right) | `audio_channel: 1` |

## Avoiding Leakage

**Symptom:** Hearing the other participant in audio playback when analyzing one speaker.

**Cause:** Using `mono=True` which averages both stereo channels.

**Solution:** Always use `load_audio(file, channel=X)` where X is the specific channel (0 or 1).

## Verification

To verify correct channel loading:

```python
import numpy as np
from utils.io import load_audio

# Load both channels separately
interviewer, sr = load_audio('console_mic_B1.wav', channel=0)
participant, sr = load_audio('console_mic_B1.wav', channel=1)

# Verify they're different (not averaged)
assert not np.allclose(interviewer, participant), "Channels should be different!"

# Check energy distribution
print(f"Interviewer RMS: {np.sqrt(np.mean(interviewer**2)):.4f}")
print(f"Participant RMS: {np.sqrt(np.mean(participant**2)):.4f}")
```

---

**Last updated:** 2024-11-10
**Issue:** Channel leakage in audio playback
**Status:** FIXED - All modules updated to use channel-specific loading
