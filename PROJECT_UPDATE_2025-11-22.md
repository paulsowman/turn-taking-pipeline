# Project Update: Turn-Taking Pipeline Batch Processing
**Date:** 2025-11-22
**Session:** Continuation - Batch Processing Configuration

---

## Summary

Configured and debugged the batch processing pipeline (`process_all_subjects.sh`) for full end-to-end processing of all 28 subjects with interviewer speech analysis, including the new stratified TRF analysis for turn-taking hypothesis testing.

---

## Key Accomplishments

### 1. Fixed Batch Processing Script Configuration ✓

**Issue:** Script was configured for sub-01 to sub-10 only, needed to handle all 28/32 subjects

**Solution:**
- Modified script to loop through sub-01 to sub-32
- Automatically detects available subjects from `outputs/sync/` directory
- Skips missing subjects (sub-12, sub-20, sub-28, sub-30) gracefully
- Shows progress counter: "PROCESSING sub-XX (N/28)"

**Files Modified:**
- `process_all_subjects.sh` - Dynamic subject detection
- Fixed configuration: `SPEAKER="interviewer"` (no longer a parameter)

### 2. Fixed Conda Environment Activation ✓

**Issue:** MFA step was failing with "MFA not installed" error when using `conda run -n mfa`

**Root Cause:** `conda run` doesn't fully activate the environment, so subprocess calls to `mfa` command fail

**Solution:** Reverted to proper conda activation:
```bash
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
python scripts/run_mfa_alignment.py ...
conda deactivate
```

**Files Modified:**
- `process_all_subjects.sh` - Step 3 (MFA alignment)
- `process_single_subject.sh` - Step 3 (MFA alignment)

**Commit:** `8b99a21` - Fix conda environment activation for MFA step

### 3. Added Robust Error Handling ✓

**Issue:** Script would crash on first error, preventing batch processing of all subjects

**Solution:** Added error handling to continue processing even when individual analyses fail:

```bash
python scripts/analyze_trf_combined.py ... || {
    echo "⚠ Warning: TRF analysis failed for $SUBJECT"
    echo "  Continuing to next step..."
}
```

**Applied to:**
- Step 7: TRF analysis (might fail if nursery_rhyme missing)
- Step 8: Turn-taking exploration (diagnostic)
- Step 9: Stratified TRF analysis

**Files Modified:**
- `process_all_subjects.sh` - Error handling for Steps 7-9
- `scripts/analyze_trf_combined.py` - Graceful handling of missing nursery_rhyme

**Commit:** `de3aa62` - Add stratified analysis to batch processing script

### 4. Made TRF Analysis Resilient to Missing Data ✓

**Enhancement:** Script now handles subjects with incomplete data gracefully

**Changes to `analyze_trf_combined.py`:**
```python
# Check if nursery_rhyme file exists before loading
if not os.path.exists(nursery_file):
    print(f"\n⚠ Warning: Nursery rhyme condition not found for {subject}")
    return trf_conv, None

# Wrap analysis in try-except
try:
    trf_nursery = analyze_condition(...)
except Exception as e:
    print(f"\n⚠ Warning: Failed to analyze nursery_rhyme: {e}")
    return trf_conv, None
```

**Result:**
- Conversation analysis completes even if nursery_rhyme data is missing
- Clear warning messages indicate what was skipped
- Batch processing continues to next subject

---

## Current Pipeline Configuration

### Batch Processing Settings

```bash
SPEAKER="interviewer"          # Fixed: analyzing interviewer speech only
RUNS=(1 2 3 4 5)              # All 5 runs per subject
DOWNSAMPLE=100                 # 100 Hz for 10x TRF fitting speedup
OVERWRITE=true                 # Fresh regeneration of all files
```

### Pipeline Steps (9 Total)

1. **Whisper Transcription** - Dual-speaker ASR with word timestamps
2. **MFA Format Conversion** - Prepare input for forced alignment
3. **Forced Alignment** - Precise word/phone boundaries (MFA)
4. **Audio-MEG Sync** - Time alignment between audio and MEG
5. **Create TRF Predictors** - Generate 23 predictors @ 100Hz
6. **Combine Runs** - Concatenate by condition (conversation, nursery_rhyme)
7. **TRF Analysis** - Fit models for both conditions, compare
8. **Explore Turn-Taking** - Diagnostic visualization (optional)
9. **Stratified TRF** - Test turn-taking hypothesis (FAR vs CLOSE)

### TRF Predictors (23 Total)

**Basic (13):**
1-2. Envelope (interviewer, participant)
3-4. Envelope (MEG channels)
5-6. F0 (interviewer, participant)
7-8. Word onsets (interviewer, participant)
9-10. Surprisal (interviewer, participant)
11-12. Duration (interviewer, participant)
13. Speaker (binary mask)

**Prosodic Deviations (6):**
14-15. F0 deviation (interviewer, participant)
16-17. Duration deviation (interviewer, participant)
18-19. Pause (interviewer, participant)

**Turn-Taking (4):**
20-21. Distance-to-turn (interviewer, participant)
22-23. Proportion-through-turn (interviewer, participant)

---

## Issues Discovered

### Sub-03 Run-02 Failure ⚠

**Symptom:**
```
✗ Interviewer MFA transcript not found:
  outputs/features/sub-03/run-02/transcript_mfa_interviewer.csv
```

**Impact:**
- Run 2 TRF predictors not created
- Nursery_rhyme condition incomplete (needs runs 2 & 4)
- TRF comparison skipped, only conversation analyzed

**Status:** Under investigation

**Debug Actions:**
1. Created `debug_sub03.sh` to check file status for all runs
2. Need to verify audio files exist for Block 2
3. Need to check Whisper/MFA logs for errors

**Possible Causes:**
- Missing/corrupted audio file (console_mic_B2.wav or subject_mic_B2.wav)
- MFA timeout or crash during alignment
- Empty/very short Whisper transcript
- Permissions issue accessing OneDrive files

**Next Steps:**
1. Run `./debug_sub03.sh` to see which step failed
2. Manually run each pipeline step for sub-03 run-02 to isolate failure point
3. Check if pattern repeats for other subjects (run-02 or run-04 failures)

---

## Subject Status

**Total Subjects:** 28 out of 32
**Available:** sub-01 through sub-32 (excluding sub-12, sub-20, sub-28, sub-30)

**Detected From:** `outputs/sync/` directory (subjects with completed sync data)

**Processing Status:**
- ✓ Script configured to process all 28 subjects automatically
- ✓ Error handling in place to continue even with failures
- ⚠ Sub-03 run-02 issue needs resolution (may affect other subjects)

---

## Output Directory Structure

```
outputs/
├── sync/{subject}/run-{run}/              # Step 4: Sync parameters
├── trf/{subject}/run-{run}/               # Step 5: Individual run TRF FIFs
├── trf_combined/{subject}/                # Step 6: Combined conditions
│   ├── {subject}_conversation_trf_raw.fif
│   └── {subject}_nursery_rhyme_trf_raw.fif
├── trf_analysis/{subject}/                # Step 7: TRF results
│   ├── conversation/interviewer/          # Conversation TRF
│   └── nursery_rhyme/interviewer/         # Nursery rhyme TRF
├── trf_analysis/{subject}/                # Step 9: Stratified results
│   └── conversation_stratified_interviewer/
└── diagnostics/                           # Step 8: Turn-taking diagnostics
```

---

## Git Commits (This Session)

1. `089deaf` - Fix subject detection to use outputs/sync directory
2. `8b99a21` - Fix conda environment activation for MFA step
3. `de3aa62` - Add stratified analysis to batch processing script
4. `6e6c9d9` - Add debug script to check sub-03 file status (unpushed - git remote unavailable)

**Branch:** `claude/codebase-explanation-011CV15fkLvZ3QQxivhUSF9y`

---

## Documentation Updates

### Updated Files
- `PIPELINE.md` - Added Steps 8-9 (stratified analysis)
- `TODO.md` - Created with 4 research questions
- `process_all_subjects.sh` - Complete batch processing
- `process_single_subject.sh` - Single subject processing

### Scripts Modified
- `scripts/analyze_trf_combined.py` - Error handling for missing conditions
- `scripts/analyze_trf_stratified.py` - Fixed listener predictor bug (from previous session)

### Scripts Created
- `debug_sub03.sh` - Diagnostic tool for checking pipeline output status

---

## Performance Optimization

**100 Hz Downsampling:**
- Original: 1000 Hz MEG data
- Downsampled: 100 Hz for TRF fitting
- **Speedup:** ~10x faster
- **Accuracy:** Minimal impact (from Lalor et al., PLOS Comp Bio)

**Estimated Processing Time per Subject:**
- Steps 1-4 (preprocessing): ~5-10 minutes
- Step 5 (TRF predictors): ~2-3 minutes (with 100Hz)
- Steps 6-9 (analysis): ~2-5 minutes
- **Total:** ~10-20 minutes per subject
- **All 28 subjects:** ~5-10 hours total

---

## Research Focus: Turn-Taking Hypothesis

**Hypothesis:** Surprisal sensitivity decreases as listeners prepare to take their turn

**Method:** Stratified TRF analysis
- **FAR condition:** First half of turns (proportion < 0.5)
- **CLOSE condition:** Second half of turns (proportion >= 0.5)
- Compare surprisal TRF kernels between conditions

**Critical Fix (Previous Session):**
- Must use **listener's** proportion predictor, not speaker's
- When analyzing interviewer speech → use participant's proportion
- Ensures proper alignment between word onsets and turn position

**Status:**
- ✓ Implementation complete and tested
- ✓ Bug fixes validated (listener predictor)
- ✓ Added to batch processing pipeline
- ⚠ Awaiting full batch processing results

---

## Next Actions

### Immediate (Debug Sub-03)
1. Run `./debug_sub03.sh` to diagnose file status
2. Check audio file accessibility for Block 2
3. Manually run Steps 1-3 for sub-03 run-02 with verbose output
4. Check if other subjects have similar run-02/run-04 failures

### Short-term (Batch Processing)
1. Resolve sub-03 run-02 issue
2. Start full batch processing: `./process_all_subjects.sh`
3. Monitor for failures and collect error logs
4. Validate outputs for complete subjects

### Medium-term (Analysis)
1. Review stratified TRF results across subjects
2. Implement control analysis (from TODO.md):
   - Check predictor value distributions (FAR vs CLOSE)
   - Verify no systematic confounds
3. Prepare group-level analysis scripts

### Long-term (Research Questions)
See `TODO.md` for 4 prioritized research questions:
1. Control for predictor value changes (HIGH)
2. Eelbrain classification (MEDIUM)
3. Absolute time confounds (MEDIUM-HIGH)
4. Source space reconstruction (MEDIUM)

---

## Technical Notes

### Conda vs Venv Switching
The pipeline switches between Python environments:
- **venv:** Most steps (Whisper, sync, TRF analysis)
- **conda (mfa):** Step 3 only (MFA requires specific dependencies)

**Critical:** Must use proper activation, not `conda run -n`:
```bash
# Proper activation (works)
eval "$(conda shell.bash hook)"
conda activate mfa

# conda run (doesn't work for subprocess calls)
conda run -n mfa python script.py  # ✗ MFA command not found
```

### Error Handling Strategy
```bash
# Continue on error, show warning
command || {
    echo "⚠ Warning: command failed"
    echo "  Continuing..."
}

# Still respects set -e for critical errors
# But allows graceful degradation for optional steps
```

---

## Questions for User

1. **Sub-03 run-02:** Should we investigate all subjects for similar patterns before batch processing?
2. **Missing nursery_rhyme:** Is conversation-only analysis acceptable for subjects with incomplete data?
3. **Batch processing timing:** Ready to start overnight run, or debug sub-03 first?

---

## Session Context

**Continued From:** Previous TRF analysis session (stratified analysis implementation)

**Session Focus:** Batch processing configuration and robustness

**User Goal:** Complete end-to-end processing of all 28 subjects with:
- Interviewer speech analysis only
- Full regeneration (all --overwrite flags)
- Both conversation and nursery_rhyme conditions
- Stratified turn-taking analysis

**Current State:** Pipeline configured and tested, debugging sub-03 run-02 failure before full batch run
