#!/bin/bash
# Process all subjects through the complete TRF pipeline
# Fixed configuration: speaker=interviewer with full overwrite
# Usage: ./process_all_subjects.sh

set -e  # Exit on error

# Configuration
SPEAKER="interviewer"  # Fixed: analyzing interviewer speech (participant listening)
RUNS=(1 2 3 4 5)
VENV_PATH="venv"  # Adjust if your venv is elsewhere
CONDA_ENV="mfa"   # Adjust if your conda env has a different name
DATA_DIR="data"   # Root data directory to check for subject existence

echo "========================================================================"
echo "TURN-TAKING PIPELINE - PROCESSING ALL SUBJECTS (FRESH REDO)"
echo "========================================================================"
echo "Speaker: $SPEAKER (analyzing interviewer speech)"
echo "Subject range: sub-01 to sub-32 (will skip missing subjects)"
echo "Runs per subject: ${RUNS[@]}"
echo "venv: $VENV_PATH"
echo "conda env: $CONDA_ENV"
echo ""
echo "WARNING: --overwrite flags are SET - this will regenerate all files!"
echo ""

# Count available subjects
SUBJECT_COUNT=0
for i in {1..32}; do
    SUBJECT=$(printf "sub-%02d" $i)
    if [ -d "$DATA_DIR/$SUBJECT" ]; then
        SUBJECT_COUNT=$((SUBJECT_COUNT + 1))
    fi
done
echo "Found $SUBJECT_COUNT subjects in $DATA_DIR/"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Process subjects sub-01 through sub-32, skipping missing ones
PROCESSED_COUNT=0
SKIPPED_COUNT=0

for i in {1..32}; do
    SUBJECT=$(printf "sub-%02d" $i)

    # Check if subject directory exists
    if [ ! -d "$DATA_DIR/$SUBJECT" ]; then
        echo ""
        echo "⊘ Skipping $SUBJECT (directory not found)"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi

    echo ""
    echo "========================================================================"
    echo "PROCESSING $SUBJECT ($((PROCESSED_COUNT + 1))/$SUBJECT_COUNT)"
    echo "========================================================================"

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
    python scripts/create_trf_fif.py --subject "$SUBJECT" --runs 1 2 3 4 5 --downsample 100 --overwrite

    # Step 6: Combine runs by condition
    echo ""
    echo "Step 6: Combine runs by condition"
    python scripts/combine_runs.py --subject "$SUBJECT" --overwrite

    # Step 7: TRF analysis (with parameters for 0ms visualization start)
    echo ""
    echo "Step 7: TRF analysis (speaker: $SPEAKER)"
    python scripts/analyze_trf_combined.py "$SUBJECT" --compare --speaker "$SPEAKER" \
        --tstart -0.2 --tstop 0.6 --crop-start -0.1 --crop-stop 0.55

    # Step 8: Turn-taking predictor exploration (optional diagnostic)
    echo ""
    echo "Step 8: Explore turn-taking predictors (diagnostic)"
    # Only run for conversation condition and specific speaker (not both)
    if [[ "$SPEAKER" != "both" ]]; then
        echo "  Visualizing distance and proportion predictors for conversation..."
        python scripts/explore_distance_to_turn.py --subject "$SUBJECT" --condition conversation --speaker "$SPEAKER"
    else
        echo "  Skipping (speaker='both' - run separately for interviewer/participant)"
    fi

    # Step 9: Stratified TRF analysis (turn-taking hypothesis)
    echo ""
    echo "Step 9: Stratified TRF analysis (turn-taking hypothesis)"
    # Only run for conversation condition and specific speaker (not both)
    if [[ "$SPEAKER" != "both" ]]; then
        echo "  Testing surprisal sensitivity modulation for conversation..."
        python scripts/analyze_trf_stratified.py "$SUBJECT" --condition conversation --speaker "$SPEAKER" \
            --tstart -0.2 --tstop 0.6
    else
        echo "  Skipping (speaker='both' - run separately for interviewer/participant)"
    fi

    deactivate

    echo ""
    echo "✓ $SUBJECT complete!"
    echo ""

    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
done

echo ""
echo "========================================================================"
echo "ALL SUBJECTS PROCESSED SUCCESSFULLY"
echo "========================================================================"
echo "Processed: $PROCESSED_COUNT subjects"
echo "Skipped: $SKIPPED_COUNT subjects (not found)"
echo ""
echo "Results:"
echo "  - Basic TRF: outputs/trf_analysis/{subject}/"
if [[ "$SPEAKER" != "both" ]]; then
    echo "  - Stratified TRF: outputs/trf_analysis/{subject}/conversation_stratified_$SPEAKER/"
    echo "  - Diagnostics: outputs/diagnostics/"
fi
echo ""
