#!/bin/bash
# Process all subjects through the complete TRF pipeline
# Switches between venv (for most steps) and conda (for MFA)

set -e  # Exit on error

# Configuration
SUBJECTS=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06" "sub-07" "sub-08" "sub-09" "sub-10")
RUNS=(1 2 3 4 5)
VENV_PATH="venv"  # Adjust if your venv is elsewhere
CONDA_ENV="mfa"   # Adjust if your conda env has a different name

echo "========================================================================"
echo "TURN-TAKING PIPELINE - PROCESSING ALL SUBJECTS"
echo "========================================================================"
echo "Subjects: ${SUBJECTS[@]}"
echo "Runs per subject: ${RUNS[@]}"
echo "venv: $VENV_PATH"
echo "conda env: $CONDA_ENV"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

for SUBJECT in "${SUBJECTS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "PROCESSING $SUBJECT"
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
    echo "Step 3: Forced alignment (switching to conda)"
    echo "Activating conda env: $CONDA_ENV"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"

    for RUN in "${RUNS[@]}"; do
        echo "  Processing $SUBJECT run $RUN..."
        python scripts/run_mfa_alignment.py --subject "$SUBJECT" --run "$RUN"
    done

    conda deactivate

    # Step 4: Audio-MEG synchronization
    echo ""
    echo "Step 4: Audio-MEG synchronization (switching back to venv)"
    source "$VENV_PATH/bin/activate"

    for RUN in "${RUNS[@]}"; do
        echo "  Processing $SUBJECT run $RUN..."
        python scripts/sync_audio_meg.py --subject "$SUBJECT" --run "$RUN"
    done

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
    echo "Step 7: TRF analysis"
    python scripts/analyze_trf_combined.py "$SUBJECT" --compare --speaker participant \
        --tstart -0.05 --tstop 0.6 --crop-start -0.05 --crop-stop 0.55

    deactivate

    echo ""
    echo "✓ $SUBJECT complete!"
    echo ""
done

echo ""
echo "========================================================================"
echo "ALL SUBJECTS PROCESSED SUCCESSFULLY"
echo "========================================================================"
echo "Results are in: outputs/trf_analysis/{subject}/"
echo ""
