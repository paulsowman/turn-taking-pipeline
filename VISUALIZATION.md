# TRF Predictor Visualization

Interactive HTML visualizations for TRF predictor channels with full zoom/pan capabilities.

## Two Visualization Options

### 1. Detailed Subplots (`visualize_trf_predictors.py`)

**Best for:** Detailed inspection of individual channels

Each predictor gets its own subplot with independent y-axis scaling.

```bash
# Basic usage
python scripts/visualize_trf_predictors.py --subject sub-01 --run 1

# Auto-open in browser
python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --open

# Show specific time window
python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --tmin 10 --tmax 60

# Custom output location
python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --output my_viz.html
```

**Features:**
- 19 subplots (one per predictor channel)
- Independent y-axis for each channel
- Z-scored normalization for display
- Shows original values on hover
- Color-coded by predictor type
- Grouped by speaker (Interviewer/Participant/Shared)

### 2. Simple Stacked View (`visualize_trf_predictors_simple.py`)

**Best for:** Quick overview and comparison across channels

All predictors in a single plot with vertical offsets (like EEG browser).

```bash
# Basic usage
python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 1 --open

# Adjust vertical spacing
python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 1 --spacing 6.0

# Time window
python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 1 --tmin 0 --tmax 30
```

**Features:**
- Single plot with stacked channels
- Faster rendering
- Easier to compare timing across channels
- Color-coded by speaker (Blue=Interviewer, Orange=Participant, Green=Shared)
- Adjustable vertical spacing

---

## Interactive Features

Both visualizations support:

### Zoom & Pan
- **Click and drag** on plot to zoom into a region
- **Shift + click and drag** to pan
- **Scroll wheel** on x-axis to zoom time axis
- **Double-click** to reset zoom
- **Range slider** at bottom for quick navigation

### Channel Selection
- **Click legend items** to show/hide channels
- **Double-click legend item** to isolate one channel
- **Hover** over traces to see exact values and timestamps

### Export & Save
- **Camera icon** (top right) to save as PNG
- **HTML file** is fully standalone - can email or share
- **No Python needed** to view - just open in any browser

---

## Output Location

Visualizations are saved to:
```
outputs/trf/{subject}/run-{run:02d}/visualizations/
```

Example:
```
outputs/trf/sub-01/run-01/visualizations/
├── sub-01_run-01_predictors.html           # Full recording (detailed)
├── sub-01_run-01_predictors_simple.html    # Full recording (stacked)
├── sub-01_run-01_predictors_10-60s.html    # Time window (detailed)
```

---

## Color Coding

### By Predictor Type (Detailed View)
- **Blue**: Envelopes
- **Orange**: F0 (pitch)
- **Green**: Word onsets
- **Red**: Surprisal (lexical)
- **Purple**: Duration
- **Brown**: F0 deviation (prosodic)
- **Pink**: Duration deviation (prosodic)
- **Gray**: Pause
- **Cyan**: Speaker

### By Speaker (Simple View)
- **Blue**: Interviewer channels
- **Orange**: Participant channels
- **Green**: Shared (speaker channel)

---

## Use Cases

### Inspect Specific Features

**Q:** "Are word onsets correctly aligned with surprisal?"

```bash
python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --tmin 30 --tmax 40 --open
```

Zoom in and check that surprisal/duration spikes align with word_onsets.

### Compare Lexical vs. Prosodic Surprisal

**Q:** "Do lexical surprisal and F0 deviation occur at the same times?"

```bash
python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 1 --open
```

Look for temporal patterns where:
- High `surprisal_*` (lexical unexpectedness)
- High `f0_deviation_*` (prosodic unexpectedness)
- Do they co-occur or appear independently?

### Check F0 Masking

**Q:** "Is F0 properly masked when speakers aren't talking?"

```bash
python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --tmin 100 --tmax 120 --open
```

Compare:
- `f0_interviewer` should be zero when `speaker` = 2 (participant talking)
- `f0_participant` should be zero when `speaker` = 1 (interviewer talking)
- Both can be non-zero when `speaker` = 3 (overlap)

### Find Turn Transitions

**Q:** "What prosodic features mark turn boundaries?"

```bash
python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 1 --tmin 50 --tmax 100 --open
```

Look for patterns at turn transitions (when `speaker` changes):
- Increased `pause_*` before partner's speech
- `f0_deviation_*` at turn-final words
- `duration_deviation_*` (lengthening) at turn ends

### Quality Check

**Q:** "Are there any artifacts or weird values?"

```bash
python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --open
```

Check for:
- NaN or Inf values (flat lines at extreme values)
- Unexpected spikes or discontinuities
- Channels that are all zeros (computation failures)
- Unrealistic values on hover

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'plotly'"

**Solution:**
```bash
pip install plotly
# or
conda install -c plotly plotly
```

### Issue: Visualization is very slow

**Causes & Solutions:**

1. **Too much data**: Visualizing full 6-minute recording
   ```bash
   # Solution: Use time windows
   python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --tmin 0 --tmax 60
   ```

2. **Too many subplots**: Detailed view with 19 channels
   ```bash
   # Solution: Use simple stacked view
   python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 1
   ```

3. **Browser rendering**: Large HTML file
   ```bash
   # Solution: Chrome/Firefox handle Plotly better than Safari
   # Or: Export specific time windows only
   ```

### Issue: Channel labels are cut off

**Solution:** Channel labels in detailed view are in subplot titles. If still truncated:
```bash
# Use simple view with full labels on y-axis
python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 1
```

### Issue: Can't see small features

**Solution:** Zoom in on time axis:
1. Click and drag to select region
2. Use range slider at bottom
3. Or specify time window: `--tmin 30 --tmax 40`

---

## Advanced Usage

### Batch Create Visualizations

```bash
# Create visualizations for all subjects/runs
for subject in sub-01 sub-02 sub-03; do
    for run in 1 2 3 4 5; do
        python scripts/visualize_trf_predictors.py --subject $subject --run $run
    done
done
```

### Compare Conditions

```bash
# Conversation condition (runs 1,3,5)
python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --tmin 0 --tmax 60 --output conv_sample.html

# Nursery rhyme condition (runs 2,4)
python scripts/visualize_trf_predictors.py --subject sub-01 --run 2 --tmin 0 --tmax 60 --output nursery_sample.html

# Open both and compare prosodic variability visually
```

### Share with Collaborators

```bash
# Create standalone HTML (no dependencies)
python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 1 --output share.html

# Email share.html - recipient just opens in browser
# All interactivity preserved!
```

---

## Tips & Best Practices

### 1. Start with Simple View
Get an overview with the stacked view first, then use detailed view to inspect specific channels.

### 2. Use Time Windows
Don't visualize full 6-minute recordings - pick interesting 30-60s windows:
- Turn-rich segments
- High prosodic variability regions
- Specific linguistic phenomena

### 3. Hover for Values
Visualization shows z-scored data for display, but hover shows original values.

### 4. Compare Speakers
Toggle interviewer/participant channels in legend to compare timing and variability.

### 5. Check Correlations
Look for temporal co-occurrence patterns:
- Do `word_onsets` and `surprisal` spikes align?
- Do `f0_deviation` peaks coincide with `duration_deviation`?
- Are `pause` values high at turn boundaries?

---

## Example Workflows

### Workflow 1: Validate TRF Predictors

```bash
# Step 1: Overview
python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 1 --open

# Step 2: Check word alignment (zoom in on 30-40s)
python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --tmin 30 --tmax 40 --open

# Step 3: Verify F0 masking works (check during participant-only speech)
# Look for time when speaker=2, verify f0_interviewer is zero

# Step 4: Check prosodic deviations are reasonable
# Hover over f0_deviation spikes - should be typically -3 to +3 z-scores
```

### Workflow 2: Explore Turn-Taking

```bash
# Step 1: Find interesting turn sequence
python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 1 --open
# Scroll through and find region with multiple turns

# Step 2: Zoom in on that region
python scripts/visualize_trf_predictors.py --subject sub-01 --run 1 --tmin 75 --tmax 90 --open

# Step 3: Analyze prosodic cues at turn boundaries
# - Is there f0_deviation at turn-final words?
# - Is there duration_deviation (lengthening)?
# - Are pause values high at boundaries?
```

### Workflow 3: Compare Conditions

```bash
# Visualize same time window from both conditions
python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 1 --tmin 0 --tmax 60 --output conv.html
python scripts/visualize_trf_predictors_simple.py --subject sub-01 --run 2 --tmin 0 --tmax 60 --output nursery.html

# Open both in browser tabs and compare:
# - Conversation: High prosodic variability, irregular turn timing
# - Nursery rhyme: Low prosodic variability, predictable timing
```

---

**Pro tip:** These visualizations are perfect for presentations and papers - export as PNG at specific time windows to show example data!
