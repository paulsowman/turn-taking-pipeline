#!/bin/bash
# Debug script to check sub-03 run-02 file status and MFA alignment issues
# This script helps diagnose MFA (Montreal Forced Aligner) alignment failures for sub-03 run-02

set -e

SUBJECT="sub-03"
GROUP="G03"
RUN="02"
RUN_NUM="2"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "DEBUG SCRIPT FOR ${SUBJECT} RUN-${RUN} - MFA ALIGNMENT DIAGNOSTICS"
echo "========================================================================"
echo ""

# Base directories (from config.yaml)
MEG_BASE="/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/natural-conversations-bids/derivatives/mne-bids-pipeline-20240628"
AUDIO_BASE="/Users/em18033/Library/CloudStorage/OneDrive-AUTUniversity/Projects/Conversational_AI/Archive/data/audios"
TRANSCRIPT_BASE="/Users/em18033/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Natural_Conversations_study - Documents/analysis/audios_metadata_labelled"
OUTPUT_BASE="./outputs"

# File paths
MEG_FILE="${MEG_BASE}/${SUBJECT}/meg/${SUBJECT}_task-conversation_run-${RUN}_proc-clean_raw.fif"
AUDIO_INTERVIEWER="${AUDIO_BASE}/${GROUP}/console_mic_B${RUN_NUM}.wav"
AUDIO_PARTICIPANT="${AUDIO_BASE}/${GROUP}/subject_mic_B${RUN_NUM}.wav"
TRANSCRIPT_FILE="${TRANSCRIPT_BASE}/${GROUP}_B${RUN_NUM}.csv"
SYNC_DIR="${OUTPUT_BASE}/sync/${SUBJECT}/run-${RUN}"
FEATURES_DIR="${OUTPUT_BASE}/features/${SUBJECT}/run-${RUN}"
SYNC_PARAMS="${SYNC_DIR}/sync_params.json"
SYNC_QC="${SYNC_DIR}/sync_qc_report.txt"

echo "CHECKING FILE EXISTENCE:"
echo "------------------------------------------------------------------------"

# Check MEG file
if [ -f "$MEG_FILE" ]; then
    echo -e "${GREEN}✓${NC} MEG file exists"
    MEG_SIZE=$(ls -lh "$MEG_FILE" | awk '{print $5}')
    echo "  Size: ${MEG_SIZE}"
else
    echo -e "${RED}✗${NC} MEG file NOT found"
    echo "  Expected: ${MEG_FILE}"
fi
echo ""

# Check audio files
if [ -f "$AUDIO_INTERVIEWER" ]; then
    echo -e "${GREEN}✓${NC} Interviewer audio file exists"
    INT_SIZE=$(ls -lh "$AUDIO_INTERVIEWER" | awk '{print $5}')
    echo "  Size: ${INT_SIZE}"
    # Check if file is readable and has content
    if [ -r "$AUDIO_INTERVIEWER" ] && [ -s "$AUDIO_INTERVIEWER" ]; then
        echo -e "  ${GREEN}File is readable and not empty${NC}"
    else
        echo -e "  ${RED}WARNING: File may be empty or not readable${NC}"
    fi
else
    echo -e "${RED}✗${NC} Interviewer audio file NOT found"
    echo "  Expected: ${AUDIO_INTERVIEWER}"
fi
echo ""

if [ -f "$AUDIO_PARTICIPANT" ]; then
    echo -e "${GREEN}✓${NC} Participant audio file exists"
    PART_SIZE=$(ls -lh "$AUDIO_PARTICIPANT" | awk '{print $5}')
    echo "  Size: ${PART_SIZE}"
    if [ -r "$AUDIO_PARTICIPANT" ] && [ -s "$AUDIO_PARTICIPANT" ]; then
        echo -e "  ${GREEN}File is readable and not empty${NC}"
    else
        echo -e "  ${RED}WARNING: File may be empty or not readable${NC}"
    fi
else
    echo -e "${RED}✗${NC} Participant audio file NOT found"
    echo "  Expected: ${AUDIO_PARTICIPANT}"
fi
echo ""

# Check transcript file
if [ -f "$TRANSCRIPT_FILE" ]; then
    echo -e "${GREEN}✓${NC} Transcript file exists"
    TRANS_SIZE=$(ls -lh "$TRANSCRIPT_FILE" | awk '{print $5}')
    echo "  Size: ${TRANS_SIZE}"
    # Count lines in transcript
    LINE_COUNT=$(wc -l < "$TRANSCRIPT_FILE")
    echo "  Lines: ${LINE_COUNT}"
else
    echo -e "${RED}✗${NC} Transcript file NOT found"
    echo "  Expected: ${TRANSCRIPT_FILE}"
fi
echo ""

echo "SYNCHRONIZATION OUTPUT STATUS:"
echo "------------------------------------------------------------------------"

# Check sync directory
if [ -d "$SYNC_DIR" ]; then
    echo -e "${GREEN}✓${NC} Sync directory exists: ${SYNC_DIR}"
    echo "  Contents:"
    ls -lh "$SYNC_DIR" | tail -n +2 | awk '{print "    " $9, "(" $5 ")"}'
else
    echo -e "${RED}✗${NC} Sync directory NOT found: ${SYNC_DIR}"
fi
echo ""

# Display sync parameters if available
if [ -f "$SYNC_PARAMS" ]; then
    echo -e "${BLUE}SYNC PARAMETERS:${NC}"
    echo "------------------------------------------------------------------------"
    if command -v jq &> /dev/null; then
        # Pretty print with jq if available
        cat "$SYNC_PARAMS" | jq '.'
    else
        # Plain cat if jq not available
        cat "$SYNC_PARAMS"
    fi
    echo ""
fi

# Display QC report if available
if [ -f "$SYNC_QC" ]; then
    echo -e "${BLUE}QUALITY CONTROL REPORT:${NC}"
    echo "------------------------------------------------------------------------"
    cat "$SYNC_QC"
    echo ""
fi

# Check for visualization files
echo "VISUALIZATION FILES:"
echo "------------------------------------------------------------------------"
ENVELOPE_PLOT="${SYNC_DIR}/envelope_alignment.png"
WAVEFORM_PLOT="${SYNC_DIR}/waveform_alignment.png"

if [ -f "$ENVELOPE_PLOT" ]; then
    echo -e "${GREEN}✓${NC} Envelope alignment plot: ${ENVELOPE_PLOT}"
else
    echo -e "${YELLOW}!${NC} Envelope alignment plot not found"
fi

if [ -f "$WAVEFORM_PLOT" ]; then
    echo -e "${GREEN}✓${NC} Waveform alignment plot: ${WAVEFORM_PLOT}"
else
    echo -e "${YELLOW}!${NC} Waveform alignment plot not found"
fi
echo ""

# Summary
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"

# Extract key metrics if sync params exist
if [ -f "$SYNC_PARAMS" ]; then
    if command -v jq &> /dev/null; then
        OFFSET=$(jq -r '.initial_offset_s' "$SYNC_PARAMS")
        CORRELATION=$(jq -r '.qc_metrics.peak_correlation' "$SYNC_PARAMS")
        QUALITY=$(jq -r '.qc_metrics.sync_quality' "$SYNC_PARAMS")
        ERROR=$(jq -r '.qc_metrics.median_error_ms' "$SYNC_PARAMS")
        
        echo "Initial offset:        ${OFFSET} seconds"
        echo "Peak correlation:      ${CORRELATION}"
        echo "Median error:          ${ERROR} ms"
        echo "Sync quality:          ${QUALITY}"
        
        # Color-code quality assessment
        if [ "$QUALITY" = "excellent" ]; then
            echo -e "${GREEN}Status: Synchronization is EXCELLENT${NC}"
        elif [ "$QUALITY" = "good" ]; then
            echo -e "${BLUE}Status: Synchronization is GOOD${NC}"
        elif [ "$QUALITY" = "acceptable" ]; then
            echo -e "${YELLOW}Status: Synchronization is ACCEPTABLE${NC}"
        else
            echo -e "${RED}Status: Synchronization NEEDS REVIEW${NC}"
        fi
    else
        echo "Install 'jq' for detailed metric summary"
    fi
else
    echo -e "${YELLOW}No synchronization results found${NC}"
    echo "Run synchronization first using:"
    echo "  python scripts/run_audio_meg_sync.py --subject ${SUBJECT} --run ${RUN_NUM}"
fi

echo ""
# Check for MFA-related outputs and diagnostics
echo "MFA ALIGNMENT STATUS:"
echo "------------------------------------------------------------------------"

# Check if features directory exists
if [ -d "$FEATURES_DIR" ]; then
    echo -e "${GREEN}✓${NC} Features directory exists: ${FEATURES_DIR}"
    echo "  Contents:"
    ls -lh "$FEATURES_DIR" 2>/dev/null | tail -n +2 | awk '{print "    " $9, "(" $5 ")"}'
    
    # Check for specific MFA output files
    if [ -f "${FEATURES_DIR}/words_interviewer.csv" ]; then
        echo -e "  ${GREEN}✓${NC} Interviewer word alignments found"
    else
        echo -e "  ${YELLOW}!${NC} Interviewer word alignments NOT found"
    fi
    
    if [ -f "${FEATURES_DIR}/words_participant.csv" ]; then
        echo -e "  ${GREEN}✓${NC} Participant word alignments found"
    else
        echo -e "  ${YELLOW}!${NC} Participant word alignments NOT found"
    fi
    
    if [ -f "${FEATURES_DIR}/phones_interviewer.csv" ]; then
        echo -e "  ${GREEN}✓${NC} Interviewer phone alignments found"
    else
        echo -e "  ${YELLOW}!${NC} Interviewer phone alignments NOT found"
    fi
    
    if [ -f "${FEATURES_DIR}/phones_participant.csv" ]; then
        echo -e "  ${GREEN}✓${NC} Participant phone alignments found"
    else
        echo -e "  ${YELLOW}!${NC} Participant phone alignments NOT found"
    fi
else
    echo -e "${RED}✗${NC} Features directory NOT found: ${FEATURES_DIR}"
    echo "  MFA alignment has not been run yet"
fi
echo ""

# Check for MFA temporary directories
echo "MFA WORKING DIRECTORIES:"
echo "------------------------------------------------------------------------"
MFA_TEMP_BASE="${OUTPUT_BASE}/mfa_temp/${SUBJECT}/run-${RUN}"
if [ -d "$MFA_TEMP_BASE" ]; then
    echo -e "${GREEN}✓${NC} MFA temp directory exists: ${MFA_TEMP_BASE}"
    echo "  Contents:"
    find "$MFA_TEMP_BASE" -type f 2>/dev/null | head -10 | while read file; do
        echo "    $(basename $file)"
    done
else
    echo -e "${YELLOW}!${NC} MFA temp directory not found (may have been cleaned up)"
    echo "  Expected: ${MFA_TEMP_BASE}"
fi
echo ""

# Check for MFA logs
echo "MFA LOGS:"
echo "------------------------------------------------------------------------"
if [ -f "${OUTPUT_BASE}/logs/mfa_${SUBJECT}_run${RUN}.log" ]; then
    echo -e "${GREEN}✓${NC} MFA log file found"
    echo "  Last 20 lines:"
    tail -20 "${OUTPUT_BASE}/logs/mfa_${SUBJECT}_run${RUN}.log" 2>/dev/null || echo "  (Unable to read log)"
elif [ -f "${OUTPUT_BASE}/logs/mfa_alignment.log" ]; then
    echo -e "${GREEN}✓${NC} General MFA log found"
    echo "  Searching for ${SUBJECT} run-${RUN} entries..."
    grep -A5 -B5 "${SUBJECT}.*run.*${RUN}" "${OUTPUT_BASE}/logs/mfa_alignment.log" 2>/dev/null | tail -30 || echo "  No entries found"
else
    echo -e "${YELLOW}!${NC} No MFA log files found"
    echo "  Check: ${OUTPUT_BASE}/logs/"
fi
echo ""

echo "========================================================================"
echo "DIAGNOSTIC SUMMARY"
echo "========================================================================"

# Provide actionable recommendations
echo ""
echo -e "${BOLD}Common MFA Alignment Issues:${NC}"
echo ""
echo "1. Audio file format issues:"
echo "   - MFA requires 16kHz mono WAV files"
echo "   - Check with: ffprobe <audio_file>"
echo "   - Convert if needed: ffmpeg -i input.wav -ar 16000 -ac 1 output.wav"
echo ""
echo "2. Transcript format issues:"
echo "   - Ensure transcript matches audio timing"
echo "   - Check for special characters or encoding issues"
echo "   - Verify speaker labels are consistent"
echo ""
echo "3. MFA model/dictionary issues:"
echo "   - Ensure English acoustic model is installed"
echo "   - Check: mfa model list acoustic"
echo "   - Install if needed: mfa model download acoustic english_us_arpa"
echo ""
echo "4. Audio-transcript mismatch:"
echo "   - Verify transcript text matches actual audio content"
echo "   - Check synchronization offset is correct"
echo ""
echo -e "${BOLD}Next Steps:${NC}"
echo ""
if [ ! -f "$AUDIO_INTERVIEWER" ] || [ ! -f "$TRANSCRIPT_FILE" ]; then
    echo -e "${YELLOW}→ Missing required files - cannot proceed with MFA alignment${NC}"
elif [ ! -d "$FEATURES_DIR" ]; then
    echo -e "${YELLOW}→ Run MFA alignment:${NC}"
    echo "  python scripts/run_mfa_alignment.py --subject ${SUBJECT} --run ${RUN_NUM}"
else
    echo -e "${YELLOW}→ Re-run MFA alignment:${NC}"
    echo "  python scripts/run_mfa_alignment.py --subject ${SUBJECT} --run ${RUN_NUM}"
    echo ""
    echo -e "${YELLOW}→ Check audio file properties:${NC}"
    echo "  ffprobe ${AUDIO_INTERVIEWER}"
    echo ""
    echo -e "${YELLOW}→ Manually test MFA on interviewer audio:${NC}"
    echo "  mfa align --clean <audio_dir> <dict> <acoustic_model> <output_dir>"
fi

echo ""
echo "========================================================================"
