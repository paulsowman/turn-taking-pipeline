#!/bin/bash
# Process a single subject through the complete TRF pipeline
# Usage: ./process_single_subject.sh sub-02 [speaker]
# speaker: participant (default), interviewer, or both

set -e  # Exit on error

# Configuration
SUBJECT="$1"
SPEAKER="${2:-participant}"  # Default to participant if not specified
RUNS=(1 2 3 4 5)
VENV_PATH="venv"  # Adjust if your venv is elsewhere
CONDA_ENV="mfa"   # Adjust if your conda env has a different name

if [ -z "$SUBJECT" ]; then
    echo "Error: No subject specified"
    echo "Usage: ./process_single_subject.sh sub-01 [speaker]"
    echo "  speaker: participant (default), interviewer, or both"
    exit 1
fi

# Validate speaker
if [[ ! "$SPEAKER" =~ ^(participant|interviewer|both)$ ]]; then
    echo "Error: Invalid speaker '$SPEAKER'"
    echo "Valid options: participant, interviewer, both"
    exit 1
fi

echo "========================================================================"
echo "TURN-TAKING PIPELINE - PROCESSING $SUBJECT"
echo "========================================================================"
echo "Speaker: $SPEAKER"
echo "Runs: ${RUNS[@]}"
echo "venv: $VENV_PATH"
echo "conda env: $CONDA_ENV"
echo ""

# Activate venv for Steps 1-2
echo "Activating venv..."
source "$VENV_PATH/bin/activate"

# Step 1: Whisper transcription
echo ""
echo "Step 1: Whisper transcription"
for RUN in "${RUNS[@]}"; do
    echo "  Processing $SUBJECT run $RUN..."
    python scripts/transcribe_dual_speaker.py --subject "$SUBJECT" --run "$RUN"
done

# Step 2: MFA format conversion
echo ""
echo "Step 2: MFA format conversion"
for RUN in "${RUNS[@]}"; do
    echo "  Processing $SUBJECT run $RUN..."
    python scripts/whisper_to_mfa.py --subject "$SUBJECT" --run "$RUN"
done

deactivate

# Step 3: Forced alignment (needs conda env with MFA)
echo ""
echo "Step 3: Forced alignment (using conda env: $CONDA_ENV)"

for RUN in "${RUNS[@]}"; do
    echo "  Processing $SUBJECT run $RUN..."
    conda run -n "$CONDA_ENV" python scripts/run_mfa_alignment.py --subject "$SUBJECT" --run "$RUN"
done

# Step 4: Audio-MEG synchronization
echo ""
echo "Step 4: Audio-MEG synchronization (switching back to venv)"
source "$VENV_PATH/bin/activate"

echo "  Running batch sync for $SUBJECT..."
python scripts/batch_audio_meg_sync.py --subjects "$SUBJECT" --runs "${RUNS[@]}"

# Step 5: Create TRF predictors (with 100Hz downsampling)
echo ""
echo "Step 5: Create TRF predictors (100Hz downsampling)"
python scripts/create_trf_fif.py --subject "$SUBJECT" --runs 1 2 3 4 5 --downsample 100

# Step 6: Combine runs by condition
echo ""
echo "Step 6: Combine runs by condition"
python scripts/combine_runs.py --subject "$SUBJECT"

# Step 7: TRF analysis (with parameters for 0ms visualization start)
echo ""
echo "Step 7: TRF analysis (speaker: $SPEAKER)"
python scripts/analyze_trf_combined.py "$SUBJECT" --compare --speaker "$SPEAKER" \
    --tstart -0.2 --tstop 0.6 --crop-start -0.1 --crop-stop 0.55

deactivate

echo ""
echo "========================================================================"
echo "✓ $SUBJECT PROCESSING COMPLETE"
echo "========================================================================"
echo "Results are in: outputs/trf_analysis/$SUBJECT/"
echo ""
