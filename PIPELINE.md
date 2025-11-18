# Turn-Taking MEG Analysis Pipeline

Complete pipeline for TRF (Temporal Response Function) analysis of turn-taking conversations using MEG data.

## Overview

This pipeline processes MEG recordings of two-person conversations to analyze how the brain responds to linguistic and prosodic features during natural turn-taking. The analysis uses boosted regression to fit TRFs relating multiple predictors (surprisal, prosody, word onsets) to MEG responses.

---

## Pipeline Steps

### Step 1: Transcription (Whisper ASR)

**Purpose:** Automatic speech recognition to get word-level timestamps for both speakers

**Input:**
- External audio recordings (2-channel or separate files for interviewer/participant)
- MEG files (for metadata)

**Command:**
```bash
# Single subject/run
python scripts/transcribe_dual_speaker.py --subject sub-01 --run 1

# All subjects/runs
python scripts/transcribe_dual_speaker.py --all

# Specify Whisper model size (base, small, medium, large)
python scripts/transcribe_dual_speaker.py --subject sub-01 --run 1 --model base
```

**Output:**
```
outputs/features/{subject}/run-{run}/
├── transcript_interviewer.csv
└── transcript_participant.csv
```

**Columns:** `start`, `end`, `text`, `words` (JSON list with per-word timing)

**Time:** ~2-5 minutes per run

---

### Step 2: MFA Format Conversion

**Purpose:** Convert Whisper transcripts to Montreal Forced Aligner input format

**Input:**
- Whisper transcript CSVs from Step 1
- External audio files

**Command:**
```bash
# Single subject/run
python scripts/whisper_to_mfa.py --subject sub-01 --run 1

# All subjects/runs
python scripts/whisper_to_mfa.py --all
```

**Output:**
```
outputs/mfa/{subject}/run-{run}/
├── interviewer/
│   ├── interviewer_001.txt (plain text, one word per line)
│   └── interviewer_001.wav (audio for that segment)
└── participant/
    ├── participant_001.txt
    └── participant_001.wav
```

**Time:** ~1 minute per run

---

### Step 3: Forced Alignment (MFA)

**Purpose:** Get precise word and phone boundaries using acoustic-phonetic models

**Input:**
- MFA text + audio files from Step 2

**Command:**
```bash
# Single subject/run
python scripts/run_mfa_alignment.py --subject sub-01 --run 1

# All subjects/runs
python scripts/run_mfa_alignment.py --all
```

**Output:**
```
outputs/features/{subject}/run-{run}/
├── transcript_mfa_interviewer.csv
└── transcript_mfa_participant.csv
```

**Columns:** `word`, `start`, `end`, `duration`, `start_meg` (MEG timebase after sync)

**Time:** ~1-3 minutes per run

**Note:** Requires Montreal Forced Aligner installed with English US ARPA model

---

### Step 4: Audio-MEG Synchronization

**Purpose:** Align external audio to MEG timebase with drift correction

**Input:**
- MEG files (with MISC 007/008 microphone channels)
- External audio recordings
- MFA transcripts from Step 3

**Command:**
```bash
# Single subject/run
python scripts/run_audio_meg_sync.py --subject sub-01 --run 1

# All subjects/runs
python scripts/batch_audio_meg_sync.py
```

**Output:**
```
outputs/sync/{subject}/run-{run}/
├── sync_params.json (offset, correlation, drift metrics)
└── diagnostic plots (envelope alignment, correlation)
```

**Time:** ~1-2 minutes per run

**Critical:** This step updates the `start_meg` column in MFA transcripts to account for audio-MEG timing offset

---

### Step 5: Create TRF Predictor Files

**Purpose:** Generate 23 TRF predictors at MEG sampling rate and add as MISC channels

**Input:**
- MEG files (1000 Hz native sampling, or downsampled to 100 Hz)
- External audio (for envelopes, F0)
- MFA transcripts with MEG-aligned boundaries (from Steps 3+4)
- Sync parameters (from Step 4)

**Command:**
```bash
# Single subject/run with 100 Hz downsampling (RECOMMENDED)
python scripts/create_trf_fif.py --subject sub-01 --runs 1 --downsample 100

# Single subject, all runs
python scripts/create_trf_fif.py --subject sub-01 --runs 1 2 3 4 5 --downsample 100

# All subjects/runs
python scripts/create_trf_fif.py --all --downsample 100

# Skip surprisal computation (faster, for testing)
python scripts/create_trf_fif.py --subject sub-01 --runs 1 --downsample 100 --no-surprisal

# Overwrite existing files
python scripts/create_trf_fif.py --subject sub-01 --runs 1 2 3 4 5 --downsample 100 --overwrite
```

**Output:**
```
outputs/trf/{subject}/run-{run}/
└── {subject}_run-{run}_trf_raw.fif
```

**Contains:** MEG channels + 23 MISC predictor channels:

**Interviewer (11 channels):**
1. `MISC_envelope_interviewer` - Audio envelope (RMS)
2. `MISC_envelope_meg_mic7` - MEG mic envelope (sync verification)
3. `MISC_f0_interviewer` - F0 contour (normalized 0-1)
4. `MISC_word_onsets_interviewer` - Delta functions at word onsets
5. `MISC_surprisal_interviewer` - GPT-2 surprisal (conversation-aware)
6. `MISC_duration_interviewer` - Word durations (normalized)
7. `MISC_f0_deviation_interviewer` - Z-scored F0 deviations
8. `MISC_duration_deviation_interviewer` - Z-scored duration deviations
9. `MISC_pause_interviewer` - Time since last word (normalized)
10. `MISC_distance_to_turn_interviewer` - Distance to next turn boundary (seconds, capped at 5s)
11. `MISC_proportion_through_turn_interviewer` - Proportion through current turn (0-1)

**Participant (11 channels):** Same as above

**Shared (1 channel):** `MISC_speaker` (0=silence, 1=interviewer, 2=participant, 3=overlap)

**Time:** ~2-5 minutes per run (with surprisal), ~1 minute without

**Note:** Downsampling to 100 Hz is now the recommended default for ~10x speedup in TRF fitting with minimal accuracy loss.

---

### Step 6: Combine Runs by Condition

**Purpose:** Concatenate runs into condition-specific files (conversation vs nursery rhyme)

**Input:**
- TRF FIF files from Step 5 (individual runs)

**Command:**
```bash
# Single subject, both conditions
python scripts/combine_runs.py --subject sub-01

# Single subject, specific condition
python scripts/combine_runs.py --subject sub-01 --condition conversation

# All subjects
python scripts/combine_runs.py --all

# Custom gap duration between runs (default: 2.0s)
python scripts/combine_runs.py --subject sub-01 --gap 3.0
```

**Output:**
```
outputs/trf_combined/{subject}/
├── {subject}_conversation_trf_raw.fif
└── {subject}_nursery_rhyme_trf_raw.fif
```

**Time:** ~30 seconds per subject

---

### Step 7: TRF Analysis

**Purpose:** Fit Temporal Response Functions using eelbrain boosted regression

**Input:**
- Combined TRF FIF files from Step 6

#### Option A: Basic TRF Analysis (analyze_trf_combined.py)

```bash
# Single condition
python scripts/analyze_trf_combined.py sub-01 --condition conversation

# Compare both conditions
python scripts/analyze_trf_combined.py sub-01 --compare

# Specify speaker (participant, interviewer, or both)
python scripts/analyze_trf_combined.py sub-01 --compare --speaker both

# Skip saving plots (faster)
python scripts/analyze_trf_combined.py sub-01 --compare --no-plots
```

**Output:**
```
outputs/trf_analysis/{subject}/{condition}/{speaker}/
├── trf_model.pickle
├── trf_word_onsets.png (butterfly + topomap)
├── trf_surprisal.png
├── trf_f0_deviation.png
├── trf_duration_deviation.png
└── trf_pause.png
```

**Time:** ~90 minutes per condition at 1000 Hz (estimated ~9 min at 100 Hz)

**Parameters:**
- Time window: -100ms to +600ms
- Basis function: 50ms
- Cross-validation: 5-fold
- Error metric: L1 (robust to outliers)
- Selective stopping: True

#### Option B: Advanced Multi-Predictor TRF (multipredictor_trf_combined.py)

**More comprehensive analysis with drop-one contribution testing**

```bash
# Single run analysis
python scripts/multipredictor_trf_combined.py sub-01 --run 1

# Compare conditions
python scripts/multipredictor_trf_combined.py sub-01 --compare

# Custom predictors
python scripts/multipredictor_trf_combined.py sub-01 --run 1 --predictors word_onsets surprisal f0
```

#### Option C: Envelope-Only TRF (envelope_trf.py)

**Baseline analysis using audio envelope only**

```bash
python scripts/envelope_trf.py sub-01 --condition conversation
```

---

### Step 8: Stratified TRF Analysis (Turn-Taking Hypothesis)

**Purpose:** Test hypothesis that surprisal sensitivity decreases as listeners prepare to take their turn

**Input:**
- Combined TRF FIF files from Step 6 (must include proportion-through-turn predictors)

**Command:**
```bash
# Analyze interviewer speech (participant listening/preparing to speak)
python scripts/analyze_trf_stratified.py sub-01 --condition conversation --speaker interviewer

# Analyze participant speech (interviewer listening/preparing to speak)
python scripts/analyze_trf_stratified.py sub-01 --condition conversation --speaker participant

# Custom time window
python scripts/analyze_trf_stratified.py sub-01 --condition conversation --speaker interviewer \
    --tstart -0.2 --tstop 0.6

# Subset of predictors
python scripts/analyze_trf_stratified.py sub-01 --condition conversation --speaker interviewer \
    --predictors envelope surprisal word_onsets
```

**Method:**
Uses **implicit masking** approach (like speaker selection):
1. Creates duplicate predictors for each stratification level:
   - `{predictor}_far`: Active in first half of turns (proportion < 0.5)
   - `{predictor}_close`: Active in second half of turns (proportion >= 0.5)
2. Fits single TRF model with all stratified predictors (12 total for 6 base predictors)
3. Compares kernels between FAR and CLOSE conditions

**Stratification Logic:**
- When analyzing **interviewer speech** with `--speaker interviewer`:
  - Uses `proportion_through_turn_participant` (listener's perspective)
  - FAR = participant in first half of interviewer's turns
  - CLOSE = participant in second half of interviewer's turns (preparing to speak)
- Hypothesis: Surprisal TRF amplitude should be smaller in CLOSE condition

**Output:**
```
outputs/trf_analysis/{subject}/{condition}_stratified_{speaker}/
├── trf_stratified_model.pickle
├── trf_envelope_comparison.png (FAR vs CLOSE overlay + difference)
├── trf_word_onsets_comparison.png
├── trf_surprisal_comparison.png
├── trf_f0_deviation_comparison.png
├── trf_duration_deviation_comparison.png
└── trf_pause_comparison.png
```

**Comparison plots include:**
- Panel 1: Overlay of FAR (blue) and CLOSE (red) TRF kernels
- Panel 2: Difference plot (FAR - CLOSE) with shaded area

**Hypothesis test output:**
```
HYPOTHESIS TEST: SURPRISAL SENSITIVITY
======================================================================

Surprisal TRF comparison:
  FAR (first half):   Peak = 2.45e-05 at 350ms
  CLOSE (second half): Peak = 1.82e-05 at 340ms
  Reduction: 25.7%

✓ HYPOTHESIS SUPPORTED: Surprisal sensitivity is reduced when close to boundary
```

**Time:** ~9 minutes per analysis at 100 Hz

**Key Advantages:**
- No concatenation artifacts (continuous data)
- No boundary exclusion needed
- Consistent with existing pipeline design
- Direct comparison within single model

---

### Step 9: Explore Turn-Taking Predictors (Optional Diagnostic)

**Purpose:** Visualize distribution of distance and proportion predictors to guide stratification

**Command:**
```bash
# Default: first 60 seconds
python scripts/explore_distance_to_turn.py --subject sub-01 --condition conversation --speaker interviewer

# Custom time window
python scripts/explore_distance_to_turn.py --subject sub-01 --condition conversation --speaker interviewer \
    --window 0 120
```

**Output:**
```
outputs/diagnostics/
└── sub-01_conversation_interviewer_distance_to_turn.png
```

**Visualizations:**
- Left column: Distance-to-turn predictor
  - Time series with threshold lines (1.0s, 2.0s)
  - Histogram showing distribution
  - Cumulative distribution function
- Right column: Proportion-through-turn predictor
  - Time series with halfway line (0.5)
  - Histogram (perfectly balanced 50/50 split)
  - Cumulative distribution

**Statistics printed:**
```
TURN-TAKING PREDICTOR STATISTICS
======================================================================
Speaker: interviewer
Condition: conversation

--- DISTANCE-TO-TURN ---
  Mean: 2.82 seconds
  Median: 2.70 seconds

Fixed-threshold stratification (using distance):
  Close to boundary: distance < 1.0s (22.6% of data)
  Far from boundary: distance > 2.0s (59.9% of data)

--- PROPORTION-THROUGH-TURN (RECOMMENDED) ---
  Valid samples (during listening): 54524 / 112500 (48.5%)
  Mean: 0.499
  Median: 0.499

Turn-length normalized stratification (using proportion):
  First half of turns: proportion < 0.5 (50.1% of listening)
  Second half of turns: proportion >= 0.5 (49.9% of listening)
```

**Time:** <1 minute

---

### Step 10: Quick ERF Validation (Optional)

**Purpose:** Fast sanity check for timing and preprocessing (~2 minutes vs 90 for TRF)

**Command:**
```bash
# Basic word-onset ERF
python scripts/compute_word_erf.py sub-01 --condition conversation

# Split by high/low surprisal to test N400 effect
python scripts/compute_word_erf.py sub-01 --condition conversation --split-by-surprisal

# Interviewer speech
python scripts/compute_word_erf.py sub-01 --condition conversation --speaker interviewer
```

**Output:**
```
outputs/erf_analysis/{subject}/{condition}/{speaker}/
├── erf_butterfly.png (all channels + mean)
├── erf_topomap.png (spatial distribution at 50, 100, 200, 300, 400ms)
└── word_onset-ave.fif (MNE evoked object)

# With --split-by-surprisal:
├── erf_butterfly.png (high vs low surprisal side-by-side)
├── erf_topomap_difference.png (high - low difference)
├── high_surprisal-ave.fif
└── low_surprisal-ave.fif
```

**Expected components:**
- M50: ~50ms (early auditory)
- M100: ~100ms (auditory cortex)
- N400: ~300-500ms (semantic processing, stronger for high surprisal)

**Time:** ~2 minutes per analysis

---

### Step 11: Group-Level Analysis

**Purpose:** Grand average TRF across subjects, statistical testing

**Input:**
- TRF models from all subjects (Step 7)

**Command:**
```bash
python scripts/group_average_trf.py
```

**Output:**
```
outputs/trf_analysis/group_results/
├── grand_average_kernels.npy
├── group_statistics.json
└── group_plots/
```

**Time:** ~10-30 minutes

---

## Complete Pipeline Examples

### Example 1: Process Single Subject from Scratch (with Turn-Taking Analysis)

```bash
# Full pipeline for sub-01
python scripts/transcribe_dual_speaker.py --subject sub-01 --run 1 2 3 4 5
python scripts/whisper_to_mfa.py --subject sub-01
python scripts/run_mfa_alignment.py --subject sub-01
python scripts/batch_audio_meg_sync.py --subjects sub-01 --runs 1 2 3 4 5
python scripts/create_trf_fif.py --subject sub-01 --runs 1 2 3 4 5 --downsample 100
python scripts/combine_runs.py --subject sub-01
python scripts/analyze_trf_combined.py sub-01 --compare --speaker both

# Turn-taking hypothesis testing
python scripts/explore_distance_to_turn.py --subject sub-01 --condition conversation --speaker interviewer
python scripts/analyze_trf_stratified.py sub-01 --condition conversation --speaker interviewer

# Quick validation (optional)
python scripts/compute_word_erf.py sub-01 --condition conversation --split-by-surprisal
```

**Total time:** ~30-45 minutes per subject at 100 Hz

### Example 2: Regenerate TRF Files Only (e.g., add new predictors)

```bash
# If you already have transcripts/MFA/sync and just want to recreate predictors
python scripts/create_trf_fif.py --subject sub-01 --runs 1 2 3 4 5 --downsample 100 --overwrite
python scripts/combine_runs.py --subject sub-01 --overwrite
python scripts/analyze_trf_combined.py sub-01 --compare
python scripts/analyze_trf_stratified.py sub-01 --condition conversation --speaker interviewer
```

### Example 3: Process All Subjects

```bash
# Batch processing
python scripts/transcribe_dual_speaker.py --all
python scripts/whisper_to_mfa.py --all
python scripts/run_mfa_alignment.py --all
python scripts/batch_audio_meg_sync.py
python scripts/create_trf_fif.py --all --downsample 100
python scripts/combine_runs.py --all
python scripts/batch_multipredictor_trf_parallel.py
python scripts/group_average_trf.py
```

---

## Directory Structure

```
turn-taking-pipeline/
├── scripts/              # All processing scripts
├── src/                  # Shared utilities
├── config/
│   └── config.yaml       # Configuration file
├── data/                 # Raw MEG and audio (gitignored)
└── outputs/
    ├── features/         # Transcripts (Whisper + MFA)
    ├── mfa/              # MFA input/output
    ├── sync/             # Audio-MEG synchronization
    ├── trf/              # Individual run TRF FIFs
    ├── trf_combined/     # Condition-combined TRF FIFs
    ├── trf_analysis/     # TRF models and plots
    └── erf_analysis/     # ERF validation results
```

---

## Dependencies

**Python packages:**
```bash
pip install mne eelbrain librosa openai-whisper montreal-forced-aligner
pip install pandas numpy scipy matplotlib seaborn
pip install transformers torch  # For GPT-2 surprisal
```

**External tools:**
- Montreal Forced Aligner (conda install montreal-forced-aligner)
- MFA English US ARPA model

**Configuration:**
- Edit `config/config.yaml` to set:
  - MEG base directory
  - Audio file paths
  - Subject inclusion/exclusion
  - MFA model settings

---

## Key Features

### 19 TRF Predictors

**Linguistic:**
- Word onsets (delta functions)
- Surprisal (GPT-2, conversation-aware)
- Word duration

**Prosodic:**
- F0 contour (fundamental frequency)
- F0 deviation (z-scored, prosodic unexpectedness)
- Duration deviation (z-scored)
- Pause duration (time since last word)

**Acoustic:**
- Audio envelope (external + MEG-recorded)

**Categorical:**
- Speaker identity (interviewer/participant/overlap)

### TRF Model Parameters

- **Time window:** -100ms to +600ms (captures anticipatory + delayed effects)
- **Basis function:** 50ms (temporal smoothing)
- **Cross-validation:** 5-fold (robust model estimation)
- **Error metric:** L1 (robust to outliers)
- **Selective stopping:** Prevents overfitting

### Expected TRF Latencies

- **Word onsets:** ~50-150ms (auditory N1/P2)
- **Surprisal:** ~200-400ms (N400-like semantic effect)
- **F0 deviation:** ~100-200ms (auditory processing)
- **Conversation > Nursery Rhyme:** Stronger effects in natural conversation

---

## Troubleshooting

### Issue: "No module named 'numpy'" or similar import errors
**Solution:** Activate virtual environment with all dependencies installed

### Issue: MFA alignment fails
**Solution:** Check that:
1. Audio files exist and are readable
2. Transcript text files are properly formatted (one word per line)
3. MFA English model is installed: `mfa model download acoustic english_us_arpa`

### Issue: Audio-MEG sync correlation is low (<0.7)
**Solution:** Check that:
1. Correct MEG microphone channel is selected (MISC 007 for interviewer, 008 for participant)
2. External audio matches the MEG recording (same session)
3. Audio quality is sufficient (not too noisy)

### Issue: TRF fitting is very slow
**Solution:**
1. Use `--no-surprisal` flag to skip GPT-2 computation (for testing)
2. Reduce data duration by analyzing single condition instead of both
3. Wait for 100 Hz downsampling implementation (10x speedup)

### Issue: No predictors found in TRF analysis
**Solution:** Check that:
1. `create_trf_fif.py` completed successfully
2. Combined FIF files exist in `outputs/trf_combined/`
3. MISC channels are present in FIF file (check with `mne.io.read_raw_fif()`)

---

## References

- **TRF Method:** Lalor et al. (2009), Crosse et al. (2016)
- **Eelbrain:** https://eelbrain.readthedocs.io/
- **MNE-Python:** https://mne.tools/
- **Montreal Forced Aligner:** https://montreal-forced-aligner.readthedocs.io/
- **Whisper ASR:** https://github.com/openai/whisper

---

## Key Features Update Summary

### Implemented (2025-11)
- ✅ 100 Hz downsampling with `--downsample` flag (~10x speedup)
- ✅ Turn-taking predictors: distance-to-turn and proportion-through-turn
- ✅ Stratified TRF analysis for testing turn-taking hypotheses
- ✅ Implicit masking approach (no concatenation artifacts)
- ✅ Diagnostic visualization for turn-taking predictors

### Active Research Questions
See [TODO.md](TODO.md) for detailed research tasks and implementation notes.

---

**Last updated:** 2025-11-18
