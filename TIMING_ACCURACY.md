# Timing Accuracy in the Turn-Taking Pipeline

This document describes the temporal resolution and accuracy of different components in the pipeline, critical for TRF analysis with MEG data at 1000 Hz (1ms resolution).

## Summary Table

| Component | Temporal Resolution | Accuracy | Use Case |
|-----------|-------------------|----------|----------|
| MEG Recording | 1 ms (1000 Hz) | < 1 ms | Brain responses |
| Audio Recording | 0.023 ms (44100 Hz) | < 0.1 ms | Speech signal |
| Audio-MEG Sync | 1 ms | ~5-10 ms | Cross-correlation alignment |
| Whisper ASR | Variable | ±50-200 ms | Initial transcription |
| Montreal Forced Aligner | ~10 ms frames | ±10-20 ms | Precise word/phone timing |
| F0 Extraction (Praat) | 10 ms (default) | ±10-20 ms | Pitch contours |
| Envelope (RMS) | 10 ms (default) | ±5-10 ms | Amplitude envelope |

## Component Details

### 1. MEG Recording (1 ms resolution)

**Resolution**: 1000 Hz = 1 ms per sample

**Accuracy**: Hardware-limited, < 1 ms jitter

**Notes**:
- Gold standard for timing in the pipeline
- All other signals must be aligned to MEG timebase
- Auxiliary channel (MISC 007) records audio trigger for synchronization

### 2. Audio Recording (44100 Hz)

**Resolution**: 44100 Hz = ~0.023 ms per sample

**Accuracy**: < 0.1 ms

**Notes**:
- Much finer resolution than MEG
- Downsampled by various processing steps
- Stereo channels: Left=interviewer, Right=participant (console_mic)

### 3. Audio-MEG Synchronization

**Method**: Cross-correlation of audio envelopes

**Accuracy**: ~5-10 ms

**Factors affecting accuracy**:
- Quality of auxiliary channel recording
- Signal-to-noise ratio
- Correlation peak sharpness (higher correlation = better sync)

**Quality metrics** (from `sync_params.json`):
- **Excellent**: correlation > 0.8, error < 10 ms
- **Good**: correlation > 0.6, error < 20 ms
- **Acceptable**: correlation > 0.4, error < 50 ms
- **Poor**: correlation < 0.4, error > 50 ms

**Example** (sub-01 run-01):
```json
{
  "peak_correlation": 0.84,
  "sync_quality": "excellent",
  "initial_offset_s": -12.235
}
```

**Impact on TRF analysis**:
- 5-10 ms error is acceptable for most TRF analysis
- MEG responses have ~100+ ms latency, so sync error is relatively small
- More critical for early sensory responses (<50 ms)

### 4. Whisper ASR (Automatic Speech Recognition)

**Method**: Deep learning attention-based transcription

**Temporal Accuracy**: ±50-200 ms (word onset jitter)

**Why so variable?**
- Whisper is optimized for transcription accuracy, not timing
- Timestamps derived from attention weights, not acoustic alignment
- No explicit phonetic model
- Worse for fast speech, overlapping speech, short function words

**Word-level timing quality**:
- **Content words** (nouns, verbs): Better, ±50-100 ms
- **Function words** (articles, prepositions): Worse, ±100-200 ms
- **Clear speech with pauses**: Better
- **Fast/overlapping speech**: Worse

**Example jitter observation**:
If you listen to onset playback with beeps at Whisper timestamps, you'll notice:
- Some beeps are perfectly aligned with word start
- Others are 50-100 ms early or late
- Occasional outliers with 200+ ms error

**Impact on TRF analysis**:
- **Problem**: 50-200 ms jitter at 1000 Hz MEG = 50-200 sample uncertainty
- **Effect**: Blurs temporal brain responses to word onsets
- **Reduces** effect sizes for timing-dependent phenomena
- **May miss** subtle onset-related responses (<50 ms)

**When Whisper timing is sufficient**:
- Envelope tracking (broad temporal features)
- Phrase-level analysis
- Turn-taking timing (>500 ms scale)
- Exploratory analysis

**When Whisper timing is insufficient**:
- Precise word onset responses (<100 ms)
- Phoneme-level analysis
- Surprisal effects that depend on exact timing
- Comparing onset vs vowel responses

### 5. Montreal Forced Aligner (MFA) - RECOMMENDED

**Method**: HMM-based acoustic-phonetic alignment

**Temporal Accuracy**: ±10-20 ms (word boundaries), ±5-10 ms (phone boundaries)

**How it works**:
1. Takes transcript text + audio
2. Uses acoustic model trained on speech data
3. Uses pronunciation dictionary (word → phones)
4. Finds optimal alignment using Viterbi algorithm
5. Outputs word and phone boundaries

**Output format**: TextGrid files with tiers for words and phones

**Advantages over Whisper**:
- ✅ 5-10x better accuracy (10-20 ms vs 50-200 ms)
- ✅ Phone-level timing (not just words)
- ✅ Standard tool in phonetics research
- ✅ Deterministic alignment (same text = same timing)
- ✅ Works with existing Whisper transcripts

**Limitations**:
- ❌ Requires correct transcript (use Whisper first)
- ❌ Slower than Whisper (but still fast: ~1-2 min per hour of audio)
- ❌ Needs acoustic model and dictionary (downloadable)
- ❌ May struggle with non-native speech, heavy accents

**Installation** (pip):
```bash
pip install montreal-forced-aligner
```

**Download models**:
```bash
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

**Workflow in pipeline**:
```
Audio + Whisper transcript → MFA alignment → Precise word/phone times → TRF analysis
```

### 6. F0 (Pitch) Extraction

**Method**: Praat autocorrelation (via Parselmouth)

**Frame rate**: 10 ms (100 Hz, configurable)

**Accuracy**: ±10-20 ms for voicing onset detection

**Notes**:
- Frame shift determines temporal resolution
- Actual F0 estimation is very accurate (~1 Hz)
- Timing of voicing onsets limited by frame rate
- Unvoiced segments (consonants) have no F0

**Configuration**:
```python
# Current settings in prosody/features.py
frame_shift = 0.01  # 10 ms = 100 Hz
f0_min = 75.0       # Hz
f0_max = 500.0      # Hz
```

### 7. Speech Envelope (RMS Energy)

**Method**: Root-mean-square energy in sliding window

**Frame rate**: 10 ms (configurable)

**Accuracy**: ±5-10 ms for amplitude changes

**Notes**:
- Very reliable temporal marker
- Good for aligning audio to MEG
- Correlates with auditory cortex responses
- Not affected by voicing (works for all speech)

## Recommendations for TRF Analysis

### Current Setup (Whisper only)

**Suitable for**:
- ✅ Speech envelope tracking
- ✅ Phrase-level timing
- ✅ F0 contour following
- ✅ General speech vs silence
- ✅ Turn-taking dynamics (>200 ms scale)

**Limitations**:
- ❌ Precise word onset responses
- ❌ Phoneme-level analysis
- ❌ Early auditory responses (<100 ms)
- ❌ Timing-sensitive surprisal effects

### Recommended Setup (Whisper + MFA)

**Pipeline**:
1. **Whisper**: Get accurate transcription with approximate timing
2. **Manual correction** (if needed): Fix any transcription errors
3. **MFA**: Re-align transcript to audio for precise timing
4. **Use MFA timestamps** for TRF analysis

**This gives you**:
- ✅ Accurate transcription (Whisper strength)
- ✅ Precise timing (MFA strength)
- ✅ Phone-level boundaries
- ✅ Standard tool with known accuracy

**Implementation**:
- Already built into pipeline (see `src/asr/mfa_alignment.py`)
- Automatic conversion from Whisper format
- Optional: can still use Whisper timing if MFA unavailable

## Verification Methods

### 1. Onset Playback Verification

**Script**: `scripts/create_onset_playback.py`

**Method**:
- Generates audio with beep at each word onset
- Listen to verify beeps align with actual word starts
- Subjective but very effective

**Expected results**:
- **Whisper**: Some beeps ~50-100 ms off
- **MFA**: Beeps should be within ~20 ms

### 2. Visual Inspection

**Script**: `scripts/verify_sync_alignment.py`

**Method**:
- Plot envelope, F0, and word onset markers
- Visual check of alignment quality

**What to look for**:
- Word onsets should fall at envelope increases
- Consonant-initial words: onset before F0
- Vowel-initial words: onset coincides with F0

### 3. Quantitative Metrics

**Future addition**: Statistical analysis of alignment quality
- Measure distance from onset to nearest envelope peak
- Compare Whisper vs MFA timing distributions
- Cross-validation with manual annotations

## Impact on Published Research

If you use Whisper timing without forced alignment, you should:

1. **Report the limitation**: "Word onset times were estimated using Whisper with ±50-200 ms accuracy"
2. **Discuss potential impact**: "Temporal jitter may reduce effect sizes for onset-locked responses"
3. **Use conservative time windows**: Wider TRF windows to account for jitter
4. **Focus on robust effects**: Emphasize effects that survive timing uncertainty

If you use MFA, you can claim:

1. **Standard timing accuracy**: "Word boundaries determined via forced alignment (±10-20 ms)"
2. **Phone-level precision**: "Consonant and vowel onsets separated using phonetic alignment"
3. **Industry standard**: MFA is widely accepted in speech/neuroscience research

## Summary and Next Steps

### Current Status
- ✅ Whisper implemented with channel selection
- ✅ Timing accuracy: ±50-200 ms (word onsets)
- ⚠️ May limit TRF analysis precision

### Recommended Addition
- ✅ Add MFA module to pipeline
- ✅ Convert Whisper → MFA format
- ✅ Parse TextGrid → DataFrame with precise times
- ✅ Use MFA times for TRF analysis

### Expected Improvement
- **Before**: ±50-200 ms (Whisper)
- **After**: ±10-20 ms (MFA)
- **Benefit**: 5-10x better temporal precision
- **Cost**: ~1-2 minutes processing time per audio file

---

**Last updated**: 2025-11-11
**Related docs**:
- `CHANNEL_LEAKAGE_FIX.md` - Audio channel separation
- `SYNC_ALIGNMENT_FIX.md` - Audio-MEG synchronization
- `AUDIO_CHANNEL_STRUCTURE.md` - Recording setup
