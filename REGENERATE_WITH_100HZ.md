# Regenerate Pipeline with 100 Hz Downsampling

## What Changed

The pipeline now downsamples MEG data to **100 Hz** in `create_trf_fif.py` BEFORE creating predictors. This provides:
- **10x speedup**: TRF fitting ~90 min → ~9 min per condition
- **Clean predictors**: Word onsets at 10ms grid (no anti-aliasing artifacts)
- **Same quality**: 10ms resolution adequate for word-level analysis

## What You DON'T Need to Re-run

**Steps 1-4 are REUSABLE** (time-based, not sample-based):
- ✅ Transcription (Whisper) - outputs in seconds
- ✅ MFA alignment - outputs in seconds
- ✅ Audio-MEG sync - outputs in seconds

These steps do NOT need to be re-run unless there were errors.

## What You DO Need to Re-run

**Steps 5-7** (sample-based, affected by downsampling):
- ❌ Step 5: Create TRF predictors (now at 100 Hz)
- ❌ Step 6: Combine runs (reads Step 5 outputs)
- ❌ Step 7: TRF analysis (reads Step 6 outputs)

---

## Bash Commands: Regenerate from Step 5

### Single Subject (e.g., sub-01)

```bash
# Step 5: Regenerate TRF FIF files at 100 Hz (--overwrite to replace 1000 Hz versions)
python scripts/create_trf_fif.py --subject sub-01 --runs 1 2 3 4 5 --overwrite

# Step 6: Re-combine runs by condition
python scripts/combine_runs.py --subject sub-01

# Step 7: Re-run TRF analysis (now ~9 min instead of 90 min!)
python scripts/analyze_trf_combined.py sub-01 --compare --speaker participant
```

### All Subjects (Batch Processing)

```bash
# Step 5: Regenerate all TRF FIF files at 100 Hz
python scripts/create_trf_fif.py --all --overwrite

# Step 6: Re-combine all subjects
python scripts/combine_runs.py --all

# Step 7: Run TRF analysis for all subjects
# (You'll need to loop or use batch scripts)
for subject in sub-01 sub-02 sub-03; do
    python scripts/analyze_trf_combined.py $subject --compare --speaker participant
done

# Step 8: Group-level analysis (optional, for multi-subject stats)
python scripts/group_average_trf.py
```

---

## Expected Output Changes

### File Sizes
- **Before (1000 Hz):** TRF FIF files ~2-3 GB per run
- **After (100 Hz):** TRF FIF files ~200-300 MB per run (10x smaller)

### Processing Time
- **Before (1000 Hz):** ~90 min TRF fitting per condition
- **After (100 Hz):** ~9 min TRF fitting per condition (10x faster)

### Data Quality
- **Sampling rate:** 1000 Hz → 100 Hz
- **Time resolution:** 1ms → 10ms (still adequate for word-level analysis)
- **Word onset impulses:** 1-sample deltas at 1ms → 1-sample deltas at 10ms
- **No quality loss:** 10ms resolution is 20x finer than typical inter-word interval (~200ms)

---

## Verification Steps

After regenerating, check that downsampling worked:

```bash
# Check a TRF FIF file sampling rate
python -c "
import mne
raw = mne.io.read_raw_fif('outputs/trf/sub-01/run-01/sub-01_run-01_trf_raw.fif', verbose=False)
print(f'Sampling rate: {raw.info[\"sfreq\"]} Hz')
print(f'Duration: {raw.times[-1]:.1f}s')
print(f'Samples: {len(raw.times)}')
"
# Should show: Sampling rate: 100.0 Hz
```

---

## What About the Sign-Flipping Issue?

You mentioned TRFs showing opposite polarities that average to zero. **Good news: Your codebase already has solutions!**

### Solution 1: Use Group-Level Analysis (Recommended)

The script `group_average_trf.py` automatically handles polarity alignment:

```bash
# This script includes polarity alignment by default
python scripts/group_average_trf.py
```

**What it does:**
- **Majority vote per sensor**: Determines reference polarity from most subjects
- **Flips mismatched sensors**: Aligns all subjects to reference
- **Tracks statistics**: Reports % of sensors flipped (saved to JSON)
- **Dual visualization**: Shows both polarity-aligned mean AND RMS magnitude

**Output:**
```
outputs/group_trf/runs-*/conversation/
├── group_statistics.json       # Contains flip statistics
├── group_trf_kernels.png       # 4-panel plot (aligned + RMS)
└── aligned_kernels.npy         # Polarity-corrected kernels
```

### Solution 2: Single-Subject Polarity Alignment

**Location:** `scripts/group_average_trf.py` lines 240-281

**Method:** `align_sensor_polarities(kernel_data, times, m100_window=(0.08, 0.15))`

**How it works:**
1. For each sensor, find peak in M100 window (80-150ms)
2. If peak is negative, flip entire sensor kernel
3. Average across aligned sensors

**Why M100 window?**
- Focuses on principal auditory response (80-150ms)
- Avoids edge artifacts and late noise
- Principled approach for auditory TRFs

### Solution 3: RMS Magnitude (Polarity-Independent)

**Location:** `scripts/group_average_trf.py` lines 284-298

**Method:** `compute_rms_across_sensors(kernel_data)`

**Formula:** `RMS = sqrt(mean(kernel^2))`

**Advantage:**
- Completely bypasses polarity issues
- Always non-negative
- Confirms signal presence even when signed mean is near-zero

---

## Recommended Workflow

### For Single Subject Analysis:

```bash
# 1. Regenerate with 100 Hz
python scripts/create_trf_fif.py --subject sub-01 --runs 1 2 3 4 5 --overwrite
python scripts/combine_runs.py --subject sub-01

# 2. Run TRF analysis
python scripts/analyze_trf_combined.py sub-01 --compare --speaker participant

# 3. Use existing polarity-aware scripts
# Check group_average_trf.py for align_sensor_polarities() function
# Or compute RMS magnitude to bypass polarity entirely
```

### For Multi-Subject Analysis:

```bash
# 1. Regenerate all subjects with 100 Hz
python scripts/create_trf_fif.py --all --overwrite
python scripts/combine_runs.py --all

# 2. Run individual TRF analyses
for subject in sub-01 sub-02 sub-03 sub-04 sub-05; do
    python scripts/analyze_trf_combined.py $subject --compare --speaker participant
done

# 3. Run group-level analysis (includes automatic polarity alignment!)
python scripts/group_average_trf.py

# 4. Check flip statistics
cat outputs/group_trf/runs-*/conversation/group_statistics.json | grep "polarity_alignment" -A 5
```

---

## Polarity Alignment: Technical Details

### Code Location Map

| Function | File | Lines | Purpose |
|----------|------|-------|---------|
| `align_polarities()` | group_average_trf.py | 101-144 | Group-level majority vote |
| `align_sensor_polarities()` | group_average_trf.py | 240-281 | Single-subject M100 alignment |
| `compute_rms_across_sensors()` | group_average_trf.py | 284-298 | Magnitude-only (no polarity) |
| Multi-predictor alignment | multipredictor_trf_combined.py | 860-896 | Similar to group approach |

### Flip Statistics Example

After running `group_average_trf.py`, check the JSON output:

```json
{
  "envelope_trf": {
    "polarity_alignment": {
      "total_flips": 150,           // Total sensor-subject flips
      "pct_flipped": 23.5,          // % of all sensor-subject pairs
      "n_flipped_per_sensor": [...]  // Per-sensor breakdown
    }
  }
}
```

**Typical values:**
- 20-50% flipped is normal (MEG dipole orientation varies)
- Higher % suggests strong polarity variability across subjects
- 0% suggests all sensors already aligned (unlikely) or no polarity variation

### Visualization: 4-Panel Plot

The group scripts create a **2x2 grid** showing:

```
┌─────────────────────┬─────────────────────┐
│ Polarity-Aligned    │ Polarity-Aligned    │
│ Mean (Participant)  │ Mean (Interviewer)  │
│                     │                     │
│ - Signed response   │ - Signed response   │
│ - Shows structure   │ - Shows structure   │
├─────────────────────┼─────────────────────┤
│ RMS Magnitude       │ RMS Magnitude       │
│ (Participant)       │ (Interviewer)       │
│                     │                     │
│ - Always positive   │ - Always positive   │
│ - Confirms signal   │ - Confirms signal   │
└─────────────────────┴─────────────────────┘
```

**Interpretation:**
- **Top row:** If near-zero without alignment, shows strong polarity mixing
- **Bottom row:** Confirms signal is present (non-zero magnitude)
- **After alignment:** Top row should show clear TRF structure

---

## Troubleshooting

### Issue: "File already exists" error
**Solution:** Use `--overwrite` flag:
```bash
python scripts/create_trf_fif.py --subject sub-01 --runs 1 --overwrite
```

### Issue: Still seeing 1000 Hz in output
**Solution:**
1. Check you pulled latest code with downsampling implementation
2. Verify `--overwrite` flag was used
3. Delete old files manually if needed:
```bash
rm -r outputs/trf/sub-01/run-01/
rm -r outputs/trf_combined/sub-01/
# Then re-run Steps 5-6
```

### Issue: TRF still takes 90 minutes
**Solution:**
1. Verify sampling rate in combined file:
```bash
python -c "import mne; raw = mne.io.read_raw_fif('outputs/trf_combined/sub-01/sub-01_conversation_trf_raw.fif', verbose=False); print(raw.info['sfreq'])"
```
2. Should show 100.0 Hz, not 1000.0 Hz
3. If showing 1000 Hz, re-run Steps 5-6 with `--overwrite`

### Issue: Polarity alignment not working
**Solution:**
1. Check you're using `group_average_trf.py` (not `analyze_trf_combined.py` for single subject)
2. For single subject, extract and adapt the `align_sensor_polarities()` function
3. Alternatively, use RMS magnitude approach which bypasses polarity entirely

### Issue: "No module named 'eelbrain'" or similar
**Solution:** Activate virtual environment:
```bash
source venv/bin/activate  # or your venv path
pip install -r requirements.txt
```

---

## Summary Checklist

- [x] ✅ Downsampling implemented in `create_trf_fif.py` (commit 9ef67d0)
- [x] ✅ Time estimates in `analyze_trf_combined.py` are adaptive to sampling rate
- [x] ✅ Polarity alignment solutions exist in codebase (`group_average_trf.py`)
- [ ] ⬜ Re-run Step 5: `create_trf_fif.py --all --overwrite`
- [ ] ⬜ Re-run Step 6: `combine_runs.py --all`
- [ ] ⬜ Re-run Step 7: `analyze_trf_combined.py` for each subject
- [ ] ⬜ (Optional) Run Step 8: `group_average_trf.py` with automatic polarity alignment

---

**Estimated total time to regenerate:**
- **Per subject (5 runs):** ~10-15 min (Step 5) + 1 min (Step 6) + 9 min (Step 7) = **~20-25 min**
- **All subjects (15 subjects):** ~4-6 hours total

**Expected speedup for future analyses:** **10x faster** (90 min → 9 min per TRF fit)

---

**Last updated:** 2025-11-16
