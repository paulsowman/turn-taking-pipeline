#!/bin/bash
#
# Batch Multipredictor TRF Analysis Script
# Processes all subjects with:
#   1. Sync analysis for all conversation runs
#   2. BADA localizer ROI selection (using run-06)
#   3. Multipredictor TRF (envelope + F0) for both MEG and EEG
#   4. Both run combinations: 1+3+5 and 2+4
#

set -e  # Exit on error

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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --subjects SUB1 SUB2 ...  Process only specified subjects (e.g., sub-01 sub-02)"
            echo "                            If not specified, processes all subjects (sub-01 to sub-10)"
            echo "  --skip-sync               Skip synchronization analysis"
            echo "  --skip-localizer          Skip BADA localizer ROI selection"
            echo "  --skip-trf                Skip multipredictor TRF analysis"
            echo "  --help                    Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --subjects sub-01 sub-02"
            echo "  $0 --skip-sync --subjects sub-03"
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
echo "BATCH MULTIPREDICTOR TRF ANALYSIS"
echo "======================================================================"
echo ""
echo "Subjects to process: ${SUBJECTS[@]}"
echo "Skip sync: $SKIP_SYNC"
echo "Skip localizer: $SKIP_LOCALIZER"
echo "Skip TRF: $SKIP_TRF"
echo ""

# Activate virtual environment
cd "$PIPELINE_DIR"
source venv/bin/activate

# Process each subject
for SUBJECT in "${SUBJECTS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "PROCESSING $SUBJECT"
    echo "======================================================================"
    echo ""

    # Get audio group for this subject
    AUDIO_GROUP="${SUBJECT_TO_GROUP[$SUBJECT]}"
    if [ -z "$AUDIO_GROUP" ]; then
        echo "ERROR: No audio group mapping for $SUBJECT"
        continue
    fi

    SUBJECT_MEG_DIR="$BIDS_DIR/$SUBJECT/meg"
    AUDIO_DIR="$AUDIO_BASE_DIR/$AUDIO_GROUP"

    # Check if subject directory exists
    if [ ! -d "$SUBJECT_MEG_DIR" ]; then
        echo "WARNING: MEG directory not found for $SUBJECT: $SUBJECT_MEG_DIR"
        continue
    fi

    # Check if audio directory exists
    if [ ! -d "$AUDIO_DIR" ]; then
        echo "WARNING: Audio directory not found for $SUBJECT: $AUDIO_DIR"
        continue
    fi

    # Step 1: Sync analysis for conversation runs (1-5)
    if [ "$SKIP_SYNC" = false ]; then
        echo "----------------------------------------------------------------------"
        echo "Step 1: Synchronization Analysis (runs 1-5)"
        echo "----------------------------------------------------------------------"

        for RUN in 1 2 3 4 5; do
            RUN_PADDED=$(printf "%02d" $RUN)
            MEG_FILE="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-${RUN_PADDED}_proc-clean_raw.fif"

            if [ ! -f "$MEG_FILE" ]; then
                echo "  WARNING: MEG file not found for run $RUN: $MEG_FILE"
                continue
            fi

            SYNC_PARAMS="outputs/sync/${SUBJECT}/run-${RUN_PADDED}/sync_params.json"

            if [ -f "$SYNC_PARAMS" ]; then
                echo "  Run $RUN: Sync params already exist, skipping"
            else
                echo "  Run $RUN: Computing sync params..."
                python scripts/synchronize_audio_meg.py \
                    --subject "$SUBJECT" \
                    --run "$RUN" \
                    --meg-file "$MEG_FILE" \
                    || echo "  WARNING: Sync failed for run $RUN"
            fi
        done
    else
        echo "----------------------------------------------------------------------"
        echo "Step 1: Synchronization Analysis (SKIPPED)"
        echo "----------------------------------------------------------------------"
    fi

    # Step 2: BADA localizer ROI selection (using run-06)
    if [ "$SKIP_LOCALIZER" = false ]; then
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Step 2: BADA Localizer ROI Selection (run-06)"
        echo "----------------------------------------------------------------------"

        LOCALIZER_FILE="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-06_proc-clean_raw.fif"

        if [ ! -f "$LOCALIZER_FILE" ]; then
            echo "  WARNING: Localizer file not found: $LOCALIZER_FILE"
        else
            ROI_MAG="outputs/bada_localizer/$SUBJECT/roi_sensors_mag.json"
            ROI_EEG="outputs/bada_localizer/$SUBJECT/roi_sensors_eeg.json"

            if [ -f "$ROI_MAG" ] && [ -f "$ROI_EEG" ]; then
                echo "  ROI files already exist, skipping"
            else
                echo "  Computing ROI sensors from BADA localizer..."

                # MEG magnetometers
                echo "    - MEG magnetometers"
                python scripts/bada_localizer_analysis.py \
                    --subject "$SUBJECT" \
                    --meg-file "$LOCALIZER_FILE" \
                    --sensor-type mag \
                    || echo "  WARNING: BADA localizer failed for mag"

                # EEG
                echo "    - EEG"
                python scripts/bada_localizer_analysis.py \
                    --subject "$SUBJECT" \
                    --meg-file "$LOCALIZER_FILE" \
                    --sensor-type eeg \
                    || echo "  WARNING: BADA localizer failed for eeg"
            fi
        fi
    else
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Step 2: BADA Localizer ROI Selection (SKIPPED)"
        echo "----------------------------------------------------------------------"
    fi

    # Step 3: Multipredictor TRF analysis
    if [ "$SKIP_TRF" = false ]; then
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Step 3: Multipredictor TRF Analysis"
        echo "----------------------------------------------------------------------"

        ROI_MAG="outputs/bada_localizer/$SUBJECT/roi_sensors_mag.json"
        ROI_EEG="outputs/bada_localizer/$SUBJECT/roi_sensors_eeg.json"

        if [ ! -f "$ROI_MAG" ] || [ ! -f "$ROI_EEG" ]; then
            echo "  WARNING: ROI files not found, cannot proceed with TRF"
            echo "    Missing: $ROI_MAG or $ROI_EEG"
        else
            # Define MEG files for each run combination
            MEG_RUN_01="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-01_proc-clean_raw.fif"
            MEG_RUN_02="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-02_proc-clean_raw.fif"
            MEG_RUN_03="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-03_proc-clean_raw.fif"
            MEG_RUN_04="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-04_proc-clean_raw.fif"
            MEG_RUN_05="$SUBJECT_MEG_DIR/${SUBJECT}_task-conversation_run-05_proc-clean_raw.fif"

            # Check which files exist
            FILES_EXIST=true
            for FILE in "$MEG_RUN_01" "$MEG_RUN_02" "$MEG_RUN_03" "$MEG_RUN_04" "$MEG_RUN_05"; do
                if [ ! -f "$FILE" ]; then
                    echo "  WARNING: MEG file not found: $FILE"
                    FILES_EXIST=false
                fi
            done

            if [ "$FILES_EXIST" = false ]; then
                echo "  WARNING: Not all MEG files found, skipping TRF analysis"
            else
                # Runs 1+3+5 - MEG
                echo ""
                echo "  Runs 1+3+5 - MEG magnetometers"
                OUTPUT_DIR="outputs/trf_multipredictor/$SUBJECT/runs-1_3_5/mag"
                if [ -f "$OUTPUT_DIR/summary.json" ]; then
                    echo "    Already complete, skipping"
                else
                    python scripts/multipredictor_trf_combined.py \
                        --subject "$SUBJECT" \
                        --runs 1 3 5 \
                        --meg-files "$MEG_RUN_01" "$MEG_RUN_03" "$MEG_RUN_05" \
                        --sensor-type mag \
                        --roi-sensors "$ROI_MAG" \
                        || echo "    WARNING: TRF failed for runs 1+3+5 MAG"
                fi

                # Runs 1+3+5 - EEG
                echo ""
                echo "  Runs 1+3+5 - EEG"
                OUTPUT_DIR="outputs/trf_multipredictor/$SUBJECT/runs-1_3_5/eeg"
                if [ -f "$OUTPUT_DIR/summary.json" ]; then
                    echo "    Already complete, skipping"
                else
                    python scripts/multipredictor_trf_combined.py \
                        --subject "$SUBJECT" \
                        --runs 1 3 5 \
                        --meg-files "$MEG_RUN_01" "$MEG_RUN_03" "$MEG_RUN_05" \
                        --sensor-type eeg \
                        --roi-sensors "$ROI_EEG" \
                        || echo "    WARNING: TRF failed for runs 1+3+5 EEG"
                fi

                # Runs 2+4 - MEG
                echo ""
                echo "  Runs 2+4 - MEG magnetometers"
                OUTPUT_DIR="outputs/trf_multipredictor/$SUBJECT/runs-2_4/mag"
                if [ -f "$OUTPUT_DIR/summary.json" ]; then
                    echo "    Already complete, skipping"
                else
                    python scripts/multipredictor_trf_combined.py \
                        --subject "$SUBJECT" \
                        --runs 2 4 \
                        --meg-files "$MEG_RUN_02" "$MEG_RUN_04" \
                        --sensor-type mag \
                        --roi-sensors "$ROI_MAG" \
                        || echo "    WARNING: TRF failed for runs 2+4 MAG"
                fi

                # Runs 2+4 - EEG
                echo ""
                echo "  Runs 2+4 - EEG"
                OUTPUT_DIR="outputs/trf_multipredictor/$SUBJECT/runs-2_4/eeg"
                if [ -f "$OUTPUT_DIR/summary.json" ]; then
                    echo "    Already complete, skipping"
                else
                    python scripts/multipredictor_trf_combined.py \
                        --subject "$SUBJECT" \
                        --runs 2 4 \
                        --meg-files "$MEG_RUN_02" "$MEG_RUN_04" \
                        --sensor-type eeg \
                        --roi-sensors "$ROI_EEG" \
                        || echo "    WARNING: TRF failed for runs 2+4 EEG"
                fi
            fi
        fi
    else
        echo ""
        echo "----------------------------------------------------------------------"
        echo "Step 3: Multipredictor TRF Analysis (SKIPPED)"
        echo "----------------------------------------------------------------------"
    fi

    echo ""
    echo "======================================================================"
    echo "COMPLETED $SUBJECT"
    echo "======================================================================"
done

echo ""
echo "======================================================================"
echo "BATCH PROCESSING COMPLETE"
echo "======================================================================"
echo ""
echo "Summary of processed subjects:"
for SUBJECT in "${SUBJECTS[@]}"; do
    SUMMARY_135_MAG="outputs/trf_multipredictor/$SUBJECT/runs-1_3_5/mag/summary.json"
    SUMMARY_135_EEG="outputs/trf_multipredictor/$SUBJECT/runs-1_3_5/eeg/summary.json"
    SUMMARY_24_MAG="outputs/trf_multipredictor/$SUBJECT/runs-2_4/mag/summary.json"
    SUMMARY_24_EEG="outputs/trf_multipredictor/$SUBJECT/runs-2_4/eeg/summary.json"

    STATUS="$SUBJECT:"
    [ -f "$SUMMARY_135_MAG" ] && STATUS="$STATUS runs1+3+5-MAG✓" || STATUS="$STATUS runs1+3+5-MAG✗"
    [ -f "$SUMMARY_135_EEG" ] && STATUS="$STATUS runs1+3+5-EEG✓" || STATUS="$STATUS runs1+3+5-EEG✗"
    [ -f "$SUMMARY_24_MAG" ] && STATUS="$STATUS runs2+4-MAG✓" || STATUS="$STATUS runs2+4-MAG✗"
    [ -f "$SUMMARY_24_EEG" ] && STATUS="$STATUS runs2+4-EEG✓" || STATUS="$STATUS runs2+4-EEG✗"

    echo "  $STATUS"
done

echo ""
echo "All done!"
