# Turn-Taking MEG Analysis - Research TODO

Active research questions and implementation tasks for the turn-taking TRF analysis pipeline.

---

## Priority Research Questions

### 1. Control for Predictor Value Changes Across Stratification

**Question:** Does surprisal TRF reduction in CLOSE condition reflect true neural changes, or systematic differences in surprisal values themselves?

**Rationale:** If surprisal values are systematically lower near turn boundaries (e.g., more predictable endings like "thank you"), the TRF reduction might reflect input differences rather than attentional modulation.

**Implementation:**
- Calculate mean/median surprisal values in FAR vs CLOSE conditions
- Compare distributions using t-tests or KS tests
- Compute effect sizes for predictor changes vs TRF changes
- If confounded, consider:
  - Matching surprisal distributions across conditions
  - Including surprisal value as covariate
  - Analyzing residualized TRFs

**Script location:** Create `scripts/analyze_predictor_distributions.py`

```python
# Pseudocode
for predictor in ['surprisal', 'f0_deviation', 'duration_deviation']:
    far_values = predictor_data[far_mask & (predictor_data != 0)]
    close_values = predictor_data[close_mask & (predictor_data != 0)]

    # Statistical comparison
    t_stat, p_value = scipy.stats.ttest_ind(far_values, close_values)
    cohen_d = (np.mean(far_values) - np.mean(close_values)) / pooled_std

    # Compare to TRF effect size
    trf_effect_size = (trf_peak_far - trf_peak_close) / trf_std
```

**Priority:** HIGH - Critical for interpreting stratified TRF results

---

### 2. Eelbrain Classification for Condition Comparison

**Question:** Can we decode conversation vs nursery rhyme from neural responses using eelbrain's classification tools?

**Background:** Conversation should show stronger/different TRF patterns than structured nursery rhymes. Classification accuracy quantifies this difference.

**Eelbrain tools to explore:**
- `eelbrain.boosting(..., model='categorical')` - Classify conditions from TRFs
- Cross-validated decoding accuracy
- Feature importance (which predictors drive classification)

**Implementation steps:**
1. Load TRF models for both conditions (all subjects)
2. Extract TRF kernels as features
3. Train classifier: conversation vs nursery rhyme
4. Report accuracy, confusion matrix, important time windows
5. Visualize decision boundaries

**References:**
- Eelbrain docs: https://eelbrain.readthedocs.io/en/stable/generated/eelbrain.boosting.html
- Check for `eelbrain.testnd` for cluster-based permutation tests

**Priority:** MEDIUM - Interesting for methods paper

---

### 3. Absolute Time Confounds in Stratification

**Question:** Do FAR and CLOSE conditions differ in absolute time within turns, potentially confounding results?

**Concern:**
- If conversation has longer turns than nursery rhyme
- FAR condition might include more "early turn" time
- CLOSE condition might include more "late turn" time
- Systematic differences in speech rate, fatigue, or attention over turn duration

**Analysis needed:**
```python
# Compare absolute position in turns
far_times = meg_times[far_mask]
close_times = meg_times[close_mask]

# Time since turn start
far_time_since_start = compute_time_since_turn_start(far_times)
close_time_since_start = compute_time_since_turn_start(close_times)

# Compare distributions
# If confounded: match distributions or include as covariate
```

**Cross-condition comparison:**
```python
# Conversation
conv_far_duration = mean_turn_duration[far_mask, conversation]
conv_close_duration = mean_turn_duration[close_mask, conversation]

# Nursery rhyme
rhyme_far_duration = mean_turn_duration[far_mask, nursery_rhyme]
rhyme_close_duration = mean_turn_duration[close_mask, nursery_rhyme]

# Test for systematic differences
```

**Priority:** MEDIUM-HIGH - Important control for publication

---

### 4. Source Space Reconstruction

**Question:** Where in the brain do turn-taking effects originate?

**Background:** We have:
- Individual head models (anatomy)
- Forward models (source → sensor mapping)
- TRF kernels in sensor space

**Goal:** Invert TRFs to source space to localize:
- Surprisal processing regions (likely temporal cortex)
- Turn-taking modulation sites (potential prefrontal/motor regions)

**Implementation approach:**

**Option A: MNE source reconstruction**
```python
import mne

# Load forward model
fwd = mne.read_forward_solution(f'data/{subject}/forward_model-fwd.fif')

# Convert TRF kernel to Evoked object
trf_kernel_evoked = mne.EvokedArray(
    data=trf_kernel_data,  # (n_sensors, n_times)
    info=meg_info,
    tmin=tstart
)

# Compute inverse operator
inv = mne.minimum_norm.make_inverse_operator(
    meg_info, fwd, noise_cov, loose=0.2, depth=0.8
)

# Apply inverse solution
stc = mne.minimum_norm.apply_inverse(
    trf_kernel_evoked, inv, lambda2=1/9, method='dSPM'
)

# Visualize
stc.plot(subject=subject, subjects_dir=subjects_dir,
         hemi='both', time_viewer=True)
```

**Option B: Eelbrain source space**
```python
# Check if eelbrain supports source space TRFs
# May need to:
# 1. Fit TRFs in sensor space (current approach)
# 2. Project kernels to source space post-hoc
# 3. Or fit TRFs directly in source space (if supported)
```

**Steps:**
1. Verify forward models exist for all subjects
2. Compute noise covariance from baseline or empty room
3. Test on single subject, single predictor (envelope first)
4. Extend to all predictors
5. Compare FAR vs CLOSE in source space
6. Group-level source statistics

**Priority:** MEDIUM - Great for visualization and interpretation

---

## Implementation Notes

### Predictor Distribution Analysis Script

Create `scripts/analyze_predictor_distributions.py`:

**Features:**
- Load combined FIF with proportion predictor
- Extract FAR/CLOSE masks
- For each predictor:
  - Compute statistics in each condition
  - Statistical tests (t-test, effect size)
  - Visualization (histograms, boxplots)
- Save report with:
  - Numerical comparisons
  - Effect size comparisons (predictor vs TRF)
  - Diagnostic plots

**Output:**
```
outputs/diagnostics/predictor_distributions/
├── {subject}_{condition}_{speaker}_distributions.png
├── {subject}_{condition}_{speaker}_statistics.json
└── group_predictor_comparison.csv
```

---

### Source Reconstruction Pipeline

**Prerequisites:**
- Head models: Check `data/{subject}/anatomy/` or similar
- Forward models: Look for `-fwd.fif` files
- Noise covariance: Compute from pre-stimulus baseline or empty room

**Verification script:**
```bash
# Check what anatomy files exist
python scripts/check_anatomy_files.py

# Output:
# sub-01: ✓ Forward model found
# sub-01: ✓ BEM surfaces found
# sub-01: ✓ Source space found
# sub-01: ✗ Noise covariance missing
```

**Create if missing:**
```python
# Generate noise covariance from baseline
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0, baseline=None)
noise_cov = mne.compute_covariance(epochs, method='empirical')
```

---

## Timeline Estimates

| Task | Difficulty | Time Est | Priority |
|------|-----------|----------|----------|
| Predictor distribution analysis | Low | 1-2 days | HIGH |
| Eelbrain classification | Medium | 3-5 days | MEDIUM |
| Absolute time confounds | Low | 1-2 days | MEDIUM-HIGH |
| Source reconstruction | Medium-High | 1-2 weeks | MEDIUM |

---

## Related Documentation

- **Pipeline overview:** [PIPELINE.md](PIPELINE.md)
- **Stratified analysis:** `scripts/analyze_trf_stratified.py`
- **Turn-taking predictors:** `src/predictors/trf_predictors.py`
- **Exploration tool:** `scripts/explore_distance_to_turn.py`

---

## Notes from Discussion

### F0 Deviation Definition
- **F0** = fundamental frequency (pitch)
- **Deviation** = difference from speaker's mean pitch
- **Implementation:** Z-scored relative to speaker baseline
- **Captures:** Prosodic emphasis, intonation, emotional arousal, focus

### Stratification Design Choice
- **Distance-to-turn:** Absolute time (seconds) to next boundary
  - Issue: Fixed thresholds don't account for turn length variability
  - Spike at 5s ceiling creates histogram artifacts
- **Proportion-through-turn:** Relative position (0-1) within turn
  - **Preferred:** Turn-length normalized, balanced 50/50 split
  - Better for statistical power and interpretability

### Implicit Masking Approach
- Similar to speaker selection (e.g., `--speaker interviewer`)
- Create duplicate predictors: `surprisal_far`, `surprisal_close`
- Masked predictors = 0 outside their condition → contribute only to baseline
- **Advantages:** No concatenation, no artifacts, cleaner implementation

---

**Last updated:** 2025-11-18
