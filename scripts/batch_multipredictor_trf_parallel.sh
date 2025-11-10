#!/bin/bash
#
# Parallel Batch Multipredictor TRF Analysis Script
# Processes all subjects with parallelization:
#   1. Sync analysis for all conversation runs (parallel per subject)
#   2. BADA localizer ROI selection (parallel: MEG & EEG)
#   3. Multipredictor TRF (parallel: all 4 combinations per subject)
#

# Configuration
PIPELINE_DIR="/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/turn-taking-pipeline"
BIDS_DIR="/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628"
AUDIO_BASE_DIR="/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Archive/data/audios"

# Subject mapping (subject ID to audio group)
declare -A SUBJECT_TO_GROUP
SUBJECT_TO_GROUP[sub-01]="G01"
SUBJECT_TO_GROUP[sub-02]="G02"
SUBJECT_TO_GROUP[sub-03]="G03"
SUBJECT_TO_GROUP[sub-04]="G04"
SUBJECT_TO_GROUP[sub-05]="G05"
SUBJECT_TO_GROUP[sub-06]="G06"
SUBJECT_TO_GROUP[sub-07]="G07"
SUBJECT_TO_GROUP[sub-08]="G08"
SUBJECT_TO_GROUP[sub-09]="G09"
SUBJECT_TO_GROUP[sub-10]="G10"

# Parse command-line arguments
SUBJECTS=()
SKIP_SYNC=false
SKIP_LOCALIZER=false
SKIP_TRF=false
MAX_PARALLEL=4  # Default: 4 parallel jobs

while [[ $# -gt 0 ]]; do
    case $1 in
        --subjects)
            shift
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                SUBJECTS+=("$1")
                shift
            done
            ;;
        --skip-sync)
            SKIP_SYNC=true
            shift
            ;;
        --skip-localizer)
            SKIP_LOCALIZER=true
            shift
            ;;
        --skip-trf)
            SKIP_TRF=true
            shift
            ;;
        --max-parallel)
            MAX_PARALLEL=$2
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --subjects SUB1 SUB2 ...  Process only specified subjects (e.g., sub-01 sub-02)"
            echo "                            If not specified, processes all subjects (sub-01 to sub-10)"
            echo "  --skip-sync               Skip synchronization analysis"
            echo "  --skip-localizer          Skip BADA localizer ROI selection"
            echo "  --skip-trf                Skip multipredictor TRF analysis"
            echo "  --max-parallel N          Maximum parallel jobs (default: 4)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --subjects sub-01 sub-02 --max-parallel 8"
            echo "  $0 --skip-sync --max-parallel 6"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If no subjects specified, process all
if [ ${#SUBJECTS[@]} -eq 0 ]; then
    SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10)
fi

echo "======================================================================"
echo "PARALLEL BATCH MULTIPREDICTOR TRF ANALYSIS"
echo "======================================================================"
echo ""
echo "Subjects to process: ${SUBJECTS[@]}"
echo "Skip sync: $SKIP_SYNC"
echo "Skip localizer: $SKIP_LOCALIZER"
echo "Skip TRF: $SKIP_TRF"
echo "Max parallel jobs: $MAX_PARALLEL"
echo ""

# Create log directory
LOG_DIR="$PIPELINE_DIR/logs/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Activate virtual environment
cd "$PIPELINE_DIR"
source venv/bin/activate

# ==============================================================================
# Helper function: Process single subject
# ==============================================================================
process_subject() {
    local SUBJECT=$1
    local LOG_FILE="$LOG_DIR/${SUBJECT}.log"

    {
        echo "======================================================================"
        echo "PROCESSING $SUBJECT - $(date)"
        echo "======================================================================"

        # Get audio group for this subject
        AUDIO_GROUP="${SUBJECT_TO_GROUP[$SUBJECT]}"
        if [ -z "$AUDIO_GROUP" ]; then
            echo "ERROR: No audio group mapping for $SUBJECT"
            return 1
        fi

        SUBJECT_MEG_DIR="$BIDS_DIR/$SUBJECT/meg"
        AUDIO_DIR="$AUDIO_BASE_DIR/$AUDIO_GROUP"

        # Check if subject directory exists
        if [ ! -d "$SUBJECT_MEG_DIR" ]; then
            echo "WARNING: MEG directory not found for $SUBJECT: $SUBJECT_MEG_DIR"
            return 1
        fi

        if [ ! -d "$AUDIO_DIR" ]; then
            echo "WARNING: Audio directory not found for $SUBJECT: $AUDIO_DIR"
            return 1
        fi

        # Step 1: Sync analysis (parallel within subject)
        if [ "$SKIP_SYNC" = false ]; then
            echo ""
            echo "Step 1: Synchronization Analysis (runs 1-5) - $(date)"

            # Launch sync jobs in parallel (background)
            SYNC_PIDS=()
            for RUN in 1 2 3 4 5; do
                RUN_PADDED=$(printf "%02d" $RUN)
                MEG_FILE="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-${RUN_PADDED}_proc-clean_raw.fif"

                if [ ! -f "$MEG_FILE" ]; then
                    echo "  WARNING: MEG file not found for run $RUN"
                    continue
                fi

                SYNC_PARAMS="outputs/sync/${SUBJECT}/run-${RUN_PADDED}/sync_params.json"

                if [ -f "$SYNC_PARAMS" ]; then
                    echo "  Run $RUN: Sync params already exist, skipping"
                else
                    echo "  Run $RUN: Starting sync..."
                    python scripts/synchronize_audio_meg.py \
                        --subject "$SUBJECT" \
                        --run "$RUN" \
                        --meg-file "$MEG_FILE" \
                        > "$LOG_DIR/${SUBJECT}_sync_run${RUN}.log" 2>&1 &
                    SYNC_PIDS+=($!)
                fi
            done

            # Wait for all sync jobs to complete
            if [ ${#SYNC_PIDS[@]} -gt 0 ]; then
                echo "  Waiting for ${#SYNC_PIDS[@]} sync jobs to complete..."
                for pid in "${SYNC_PIDS[@]}"; do
                    wait $pid || echo "  WARNING: Sync job $pid failed"
                done
                echo "  All sync jobs complete - $(date)"
            fi
        fi

        # Step 2: BADA localizer (parallel: MEG & EEG)
        if [ "$SKIP_LOCALIZER" = false ]; then
            echo ""
            echo "Step 2: BADA Localizer ROI Selection - $(date)"

            LOCALIZER_FILE="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-06_proc-clean_raw.fif"

            if [ ! -f "$LOCALIZER_FILE" ]; then
                echo "  WARNING: Localizer file not found: $LOCALIZER_FILE"
            else
                ROI_MAG="outputs/bada_localizer/$SUBJECT/roi_sensors_mag.json"
                ROI_EEG="outputs/bada_localizer/$SUBJECT/roi_sensors_eeg.json"

                if [ -f "$ROI_MAG" ] && [ -f "$ROI_EEG" ]; then
                    echo "  ROI files already exist, skipping"
                else
                    # Run MEG and EEG localizer in parallel
                    LOCALIZER_PIDS=()

                    if [ ! -f "$ROI_MAG" ]; then
                        echo "  Starting MEG localizer..."
                        python scripts/bada_localizer_analysis.py \
                            --subject "$SUBJECT" \
                            --meg-file "$LOCALIZER_FILE" \
                            --sensor-type mag \
                            > "$LOG_DIR/${SUBJECT}_localizer_mag.log" 2>&1 &
                        LOCALIZER_PIDS+=($!)
                    fi

                    if [ ! -f "$ROI_EEG" ]; then
                        echo "  Starting EEG localizer..."
                        python scripts/bada_localizer_analysis.py \
                            --subject "$SUBJECT" \
                            --meg-file "$LOCALIZER_FILE" \
                            --sensor-type eeg \
                            > "$LOG_DIR/${SUBJECT}_localizer_eeg.log" 2>&1 &
                        LOCALIZER_PIDS+=($!)
                    fi

                    # Wait for localizer jobs
                    if [ ${#LOCALIZER_PIDS[@]} -gt 0 ]; then
                        echo "  Waiting for ${#LOCALIZER_PIDS[@]} localizer jobs..."
                        for pid in "${LOCALIZER_PIDS[@]}"; do
                            wait $pid || echo "  WARNING: Localizer job $pid failed"
                        done
                        echo "  Localizer complete - $(date)"
                    fi
                fi
            fi
        fi

        # Step 3: Multipredictor TRF (parallel: all 4 combinations)
        if [ "$SKIP_TRF" = false ]; then
            echo ""
            echo "Step 3: Multipredictor TRF Analysis - $(date)"

            ROI_MAG="outputs/bada_localizer/$SUBJECT/roi_sensors_mag.json"
            ROI_EEG="outputs/bada_localizer/$SUBJECT/roi_sensors_eeg.json"

            if [ ! -f "$ROI_MAG" ] || [ ! -f "$ROI_EEG" ]; then
                echo "  WARNING: ROI files not found, cannot proceed with TRF"
            else
                # Define MEG files
                MEG_RUN_01="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-01_proc-clean_raw.fif"
                MEG_RUN_02="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-02_proc-clean_raw.fif"
                MEG_RUN_03="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-03_proc-clean_raw.fif"
                MEG_RUN_04="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-04_proc-clean_raw.fif"
                MEG_RUN_05="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-05_proc-clean_raw.fif"

                # Check files exist
                FILES_EXIST=true
                for FILE in "$MEG_RUN_01" "$MEG_RUN_02" "$MEG_RUN_03" "$MEG_RUN_04" "$MEG_RUN_05"; do
                    if [ ! -f "$FILE" ]; then
                        echo "  WARNING: MEG file not found: $FILE"
                        FILES_EXIST=false
                    fi
                done

                if [ "$FILES_EXIST" = true ]; then
                    TRF_PIDS=()

                    # Runs 1+3+5 - MEG
                    if [ ! -f "outputs/trf_multipredictor/$SUBJECT/runs-1_3_5/mag/summary.json" ]; then
                        echo "  Starting runs 1+3+5 MEG..."
                        python scripts/multipredictor_trf_combined.py \
                            --subject "$SUBJECT" \
                            --runs 1 3 5 \
                            --meg-files "$MEG_RUN_01" "$MEG_RUN_03" "$MEG_RUN_05" \
                            --sensor-type mag \
                            --roi-sensors "$ROI_MAG" \
                            > "$LOG_DIR/${SUBJECT}_trf_135_mag.log" 2>&1 &
                        TRF_PIDS+=($!)
                    else
                        echo "  Runs 1+3+5 MEG already complete"
                    fi

                    # Runs 1+3+5 - EEG
                    if [ ! -f "outputs/trf_multipredictor/$SUBJECT/runs-1_3_5/eeg/summary.json" ]; then
                        echo "  Starting runs 1+3+5 EEG..."
                        python scripts/multipredictor_trf_combined.py \
                            --subject "$SUBJECT" \
                            --runs 1 3 5 \
                            --meg-files "$MEG_RUN_01" "$MEG_RUN_03" "$MEG_RUN_05" \
                            --sensor-type eeg \
                            --roi-sensors "$ROI_EEG" \
                            > "$LOG_DIR/${SUBJECT}_trf_135_eeg.log" 2>&1 &
                        TRF_PIDS+=($!)
                    else
                        echo "  Runs 1+3+5 EEG already complete"
                    fi

                    # Runs 2+4 - MEG
                    if [ ! -f "outputs/trf_multipredictor/$SUBJECT/runs-2_4/mag/summary.json" ]; then
                        echo "  Starting runs 2+4 MEG..."
                        python scripts/multipredictor_trf_combined.py \
                            --subject "$SUBJECT" \
                            --runs 2 4 \
                            --meg-files "$MEG_RUN_02" "$MEG_RUN_04" \
                            --sensor-type mag \
                            --roi-sensors "$ROI_MAG" \
                            > "$LOG_DIR/${SUBJECT}_trf_24_mag.log" 2>&1 &
                        TRF_PIDS+=($!)
                    else
                        echo "  Runs 2+4 MEG already complete"
                    fi

                    # Runs 2+4 - EEG
                    if [ ! -f "outputs/trf_multipredictor/$SUBJECT/runs-2_4/eeg/summary.json" ]; then
                        echo "  Starting runs 2+4 EEG..."
                        python scripts/multipredictor_trf_combined.py \
                            --subject "$SUBJECT" \
                            --runs 2 4 \
                            --meg-files "$MEG_RUN_02" "$MEG_RUN_04" \
                            --sensor-type eeg \
                            --roi-sensors "$ROI_EEG" \
                            > "$LOG_DIR/${SUBJECT}_trf_24_eeg.log" 2>&1 &
                        TRF_PIDS+=($!)
                    else
                        echo "  Runs 2+4 EEG already complete"
                    fi

                    # Wait for all TRF jobs
                    if [ ${#TRF_PIDS[@]} -gt 0 ]; then
                        echo "  Waiting for ${#TRF_PIDS[@]} TRF jobs to complete..."
                        for pid in "${TRF_PIDS[@]}"; do
                            wait $pid || echo "  WARNING: TRF job $pid failed"
                        done
                        echo "  All TRF jobs complete - $(date)"
                    fi
                fi
            fi
        fi

        echo ""
        echo "======================================================================"
        echo "COMPLETED $SUBJECT - $(date)"
        echo "======================================================================"

    } 2>&1 | tee "$LOG_FILE"
}

# Export function and variables for parallel execution
export -f process_subject
export SUBJECT_TO_GROUP
export BIDS_DIR
export AUDIO_BASE_DIR
export PIPELINE_DIR
export LOG_DIR
export SKIP_SYNC
export SKIP_LOCALIZER
export SKIP_TRF

# ==============================================================================
# Process all subjects in parallel (limited by MAX_PARALLEL)
# ==============================================================================
echo "======================================================================"
echo "STARTING PARALLEL PROCESSING"
echo "======================================================================"
echo ""

# Process subjects in parallel using background jobs with job control
SUBJECT_PIDS=()
ACTIVE_JOBS=0

for SUBJECT in "${SUBJECTS[@]}"; do
    # Wait if we've hit the parallel limit
    while [ $ACTIVE_JOBS -ge $MAX_PARALLEL ]; do
        # Check if any jobs have finished
        for i in "${!SUBJECT_PIDS[@]}"; do
            pid=${SUBJECT_PIDS[$i]}
            if ! kill -0 $pid 2>/dev/null; then
                # Job finished
                wait $pid
                unset 'SUBJECT_PIDS[i]'
                ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
            fi
        done
        # Clean up array
        SUBJECT_PIDS=("${SUBJECT_PIDS[@]}")
        sleep 5
    done

    # Start new job
    echo "Starting $SUBJECT (job $((ACTIVE_JOBS + 1))/$MAX_PARALLEL)..."
    process_subject "$SUBJECT" &
    SUBJECT_PIDS+=($!)
    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining ${#SUBJECT_PIDS[@]} subjects to complete..."
for pid in "${SUBJECT_PIDS[@]}"; do
    wait $pid
done

# ==============================================================================
# Final summary
# ==============================================================================
echo ""
echo "======================================================================"
echo "BATCH PROCESSING COMPLETE - $(date)"
echo "======================================================================"
echo ""
echo "Summary of processed subjects:"

for SUBJECT in "${SUBJECTS[@]}"; do
    SUMMARY_135_MAG="outputs/trf_multipredictor/$SUBJECT/runs-1_3_5/mag/summary.json"
    SUMMARY_135_EEG="outputs/trf_multipredictor/$SUBJECT/runs-1_3_5/eeg/summary.json"
    SUMMARY_24_MAG="outputs/trf_multipredictor/$SUBJECT/runs-2_4/mag/summary.json"
    SUMMARY_24_EEG="outputs/trf_multipredictor/$SUBJECT/runs-2_4/eeg/summary.json"

    STATUS="$SUBJECT:"
    [ -f "$SUMMARY_135_MAG" ] && STATUS="$STATUS 1+3+5-MAG✓" || STATUS="$STATUS 1+3+5-MAG✗"
    [ -f "$SUMMARY_135_EEG" ] && STATUS="$STATUS 1+3+5-EEG✓" || STATUS="$STATUS 1+3+5-EEG✗"
    [ -f "$SUMMARY_24_MAG" ] && STATUS="$STATUS 2+4-MAG✓" || STATUS="$STATUS 2+4-MAG✗"
    [ -f "$SUMMARY_24_EEG" ] && STATUS="$STATUS 2+4-EEG✓" || STATUS="$STATUS 2+4-EEG✗"

    echo "  $STATUS"
done

echo ""
echo "Logs saved to: $LOG_DIR"
echo "All done!"
