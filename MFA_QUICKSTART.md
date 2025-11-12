# Montreal Forced Aligner (MFA) - Quick Start Guide

This guide shows you how to use MFA to get precise word timing (±10-20ms) instead of Whisper's approximate timing (±50-200ms).

## Why Use MFA?

**Problem**: Whisper's word timestamps have ±50-200ms "jitter" - not precise enough for MEG TRF analysis at 1ms resolution.

**Solution**: MFA uses acoustic-phonetic alignment to get ±10-20ms accuracy (5-10x better).

**When to use MFA**:
- ✅ TRF analysis requiring precise word onset timing
- ✅ Phoneme-level brain response analysis
- ✅ Early auditory responses (<100ms)
- ✅ Timing-sensitive surprisal effects

**When Whisper is sufficient**:
- Envelope tracking (broad temporal features)
- Phrase-level timing (>200ms scale)
- Exploratory analysis

## Installation

### Option A: Conda (RECOMMENDED - Most Reliable)

MFA has compiled dependencies (Kaldi/kalpy) that work best with conda:

```bash
# 1. Create MFA environment
conda create -n mfa -c conda-forge montreal-forced-aligner python=3.10

# 2. Activate environment
conda activate mfa

# 3. Download English acoustic model and pronunciation dictionary
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# 4. Verify installation
mfa version
```

**Pros**: Reliable, handles native dependencies automatically, well-tested

**Cons**: Requires conda, separate environment from main pipeline

### Option B: Pip (Alternative - May Have Issues)

If you prefer pip or can't use conda:

```bash
# 1. Install MFA v2.2.17 (more pip-compatible than v3.x)
pip install -r requirements_mfa.txt

# This installs:
#   - montreal-forced-aligner==2.2.17
#   - praat-textgrids>=1.4.0

# 2. Download models
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# 3. Verify installation
mfa version
```

**Pros**: No conda needed, same environment as pipeline

**Cons**: May fail with kalpy compilation errors (especially Python 3.13)

**Troubleshooting**: If you get `ModuleNotFoundError: No module named '_kalpy'`, use conda instead.

## Usage Workflow

### Step 1: Run Whisper ASR (if not already done)

```bash
# Get initial transcription with approximate timing
python scripts/test_asr_prosody.py --subject sub-01 --run 1
```

This creates: `outputs/features/sub-01/run-01/transcript.csv`

### Step 2: Convert Whisper → MFA Format

```bash
# Single subject/run
python scripts/whisper_to_mfa.py --subject sub-01 --run 1

# Multiple runs
python scripts/whisper_to_mfa.py --subject sub-01 --runs 1 2 3 4 5

# All subjects
python scripts/whisper_to_mfa.py --all
```

This creates:
- `outputs/mfa/sub-01/run-01/sub-01_run-01.txt` (transcript text)
- `outputs/mfa/sub-01/run-01/sub-01_run-01.wav` (symlink to audio)

### Step 3: Run MFA Alignment

**If using conda** (recommended):
```bash
# Activate MFA environment
conda activate mfa

# Run alignment
python scripts/run_mfa_alignment.py --subject sub-01 --run 1

# Deactivate when done
conda deactivate
```

**If using pip** (same venv):
```bash
# Just run directly
python scripts/run_mfa_alignment.py --subject sub-01 --run 1
```

**Additional options**:
```bash
# Multiple runs
python scripts/run_mfa_alignment.py --subject sub-01 --runs 1 2 3 4 5

# All subjects (takes ~1-2 min per hour of audio)
python scripts/run_mfa_alignment.py --all

# Custom models (optional)
python scripts/run_mfa_alignment.py --subject sub-01 --run 1 \
    --acoustic english_mfa \
    --dictionary english_us_mfa \
    --num-jobs 8
```

This creates:
- `outputs/features/sub-01/run-01/transcript_mfa.csv` - **Use this for TRF!**
- `outputs/features/sub-01/run-01/phones_mfa.csv` - Phone-level timing
- `outputs/features/sub-01/run-01/timing_comparison.csv` - Whisper vs MFA comparison

**Processing time**: ~1-2 minutes per hour of audio

### Step 4: Use MFA Timestamps in TRF Analysis

The MFA output (`transcript_mfa.csv`) is already aligned to MEG timebase and ready to use:

```python
import pandas as pd

# Load MFA word times (precise!)
words_mfa = pd.read_csv('outputs/features/sub-01/run-01/transcript_mfa.csv')

# Columns:
#   - word: Word text
#   - start: Start time in audio (seconds)
#   - end: End time in audio (seconds)
#   - duration: Word duration
#   - start_meg: Start time in MEG timebase (seconds)
#   - end_meg: End time in MEG timebase (seconds)
#   - duration_meg: Duration in MEG time

# For TRF analysis, use start_meg for word onset times
word_onsets_meg = words_mfa['start_meg'].values  # ±10-20ms accuracy!

# Convert to MEG samples (1000 Hz)
word_onset_samples = (word_onsets_meg * 1000).astype(int)
```

## Output Files

### transcript_mfa.csv (Word-level timing)

```csv
word,start,end,duration,start_meg,end_meg,duration_meg
So,4.20,4.72,0.52,16.435,16.955,0.52
are,4.72,5.00,0.28,16.955,17.235,0.28
you,5.00,5.12,0.12,17.235,17.355,0.12
...
```

### phones_mfa.csv (Phoneme-level timing)

```csv
phone,start,end,duration,start_meg,end_meg,duration_meg
S,4.20,4.31,0.11,16.435,16.545,0.11
OW,4.31,4.72,0.41,16.545,16.955,0.41
AA,4.72,4.82,0.10,16.955,17.055,0.10
R,4.82,5.00,0.18,17.055,17.235,0.18
...
```

### timing_comparison.csv (Whisper vs MFA)

```csv
word,whisper_start,mfa_start,time_diff,abs_diff
So,4.200,4.200,0.000,0.000
are,4.720,4.720,0.000,0.000
you,5.000,5.010,-0.010,0.010
watching,5.120,5.050,0.070,0.070
...
```

Statistical summary printed:
```
Mean absolute difference: 67.3 ms
Median absolute difference: 45.1 ms
95th percentile: 156.2 ms
Errors > 50ms: 42.3%
Errors > 100ms: 18.7%
```

## Verification

### 1. Visual Check

Use the sync verification script with MFA times:

```bash
# Update verify_sync_alignment.py to load transcript_mfa.csv instead of transcript.csv
# Then run:
python scripts/verify_sync_alignment.py --subject sub-01 --run 1 --start 10 --end 40
```

Word onset markers should align precisely with envelope increases.

### 2. Onset Playback

Create audio with beeps at MFA word onsets:

```bash
# Modify create_onset_playback.py to use transcript_mfa.csv
# Then run:
python scripts/create_onset_playback.py --subject sub-01 --run 1 --start 10 --duration 30
```

Listen to verify beeps are within ~20ms of actual word starts.

### 3. Timing Statistics

Check the timing comparison CSV:

```python
import pandas as pd

comparison = pd.read_csv('outputs/features/sub-01/run-01/timing_comparison.csv')

print(f"Mean jitter: {comparison['abs_diff'].mean() * 1000:.1f} ms")
print(f"Median jitter: {comparison['abs_diff'].median() * 1000:.1f} ms")
print(f"95th percentile: {comparison['abs_diff'].quantile(0.95) * 1000:.1f} ms")

# Large errors (Whisper was way off)
large_errors = comparison[comparison['abs_diff'] > 0.1]
print(f"\nWords with >100ms error: {len(large_errors)}")
print(large_errors[['word', 'time_diff']].head(10))
```

## Troubleshooting

### Error: ModuleNotFoundError: No module named '_kalpy'

This is the most common issue with pip installation. The kalpy package (which wraps Kaldi) failed to compile.

**Solutions**:

1. **Use conda instead** (RECOMMENDED):
   ```bash
   # Uninstall pip version
   pip uninstall montreal-forced-aligner -y

   # Install with conda
   conda create -n mfa -c conda-forge montreal-forced-aligner python=3.10
   conda activate mfa
   ```

2. **Try MFA v2.2.17** (if you must use pip):
   ```bash
   pip uninstall montreal-forced-aligner kalpy -y
   pip install montreal-forced-aligner==2.2.17 praat-textgrids
   ```

3. **Check Python version**: MFA works best with Python 3.8-3.11. Python 3.13 has compatibility issues.

### MFA not found

```bash
# Make sure it's installed
pip install montreal-forced-aligner==2.2.17

# Or with conda
conda install -c conda-forge montreal-forced-aligner

# Verify
mfa version
```

### Models not found

```bash
# List available models
mfa model list acoustic
mfa model list dictionary

# Download English models
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Verify installation
mfa model inspect acoustic english_us_arpa
mfa model inspect dictionary english_us_arpa
```

### TextGrid parsing error

```bash
# Install textgrid parser
pip install praat-textgrids
```

### MFA alignment fails

Common issues:
1. **Transcript doesn't match audio**: Check that Whisper transcript is accurate
2. **Audio quality**: MFA needs clear speech (same as Whisper)
3. **Non-standard words**: MFA may fail on words not in dictionary
4. **Memory**: Try reducing `--num-jobs` if running out of memory

### Audio file not found

Make sure you ran `whisper_to_mfa.py` first to prepare the audio files:

```bash
python scripts/whisper_to_mfa.py --subject sub-01 --run 1
```

## Performance

**Processing time**:
- ~1-2 minutes per hour of audio
- Parallelized (default: 4 jobs)
- Can run overnight for all subjects

**Disk space**:
- TextGrid files: ~100-200 KB per file
- Audio files: Symlinked (no extra space)
- Total: <100 MB for all subjects

**Memory**:
- ~2-4 GB per job
- Adjust `--num-jobs` based on available RAM

## Advanced Usage

### Custom Models

Use different acoustic models or dictionaries:

```bash
# List available models
mfa model list acoustic
mfa model list dictionary

# Download alternative models
mfa model download acoustic english_mfa
mfa model download dictionary english_us_mfa

# Use in alignment
python scripts/run_mfa_alignment.py --subject sub-01 --run 1 \
    --acoustic english_mfa \
    --dictionary english_us_mfa
```

### Phone-level Analysis

Use `phones_mfa.csv` for consonant vs vowel onset analysis:

```python
import pandas as pd

phones = pd.read_csv('outputs/features/sub-01/run-01/phones_mfa.csv')

# Separate consonants and vowels
vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
consonants = set(phones['phone'].unique()) - set(vowels)

vowel_onsets = phones[phones['phone'].isin(vowels)]['start_meg'].values
consonant_onsets = phones[phones['phone'].isin(consonants)]['start_meg'].values
```

## For Publication

If using MFA in your research:

**Method description**:
> "Word onset times were determined using the Montreal Forced Aligner (MFA) v3.0,
> which provides ±10-20ms temporal accuracy via acoustic-phonetic alignment.
> Initial transcriptions were obtained using OpenAI Whisper, then re-aligned using
> the english_us_arpa acoustic model and pronunciation dictionary."

**Citation**:
> McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., & Sonderegger, M. (2017).
> Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi.
> In Interspeech (Vol. 2017, pp. 498-502).

## References

- **MFA Documentation**: https://montreal-forced-aligner.readthedocs.io/
- **GitHub**: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
- **Timing Accuracy**: See `TIMING_ACCURACY.md` in this repository

---

**Questions?** See `TIMING_ACCURACY.md` for detailed accuracy analysis and recommendations.
