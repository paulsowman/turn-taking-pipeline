#!/bin/bash
# Debug script to check sub-03 file status

SUBJECT="sub-03"

echo "=================================================================="
echo "DEBUGGING SUB-03 FILE STATUS"
echo "=================================================================="
echo ""

for RUN in 1 2 3 4 5; do
    echo "----------------------------------------"
    echo "RUN $RUN:"
    echo "----------------------------------------"

    # Check Whisper transcripts (Step 1 output)
    WHISPER_DIR="outputs/features/$SUBJECT/run-0$RUN"
    if [ -d "$WHISPER_DIR" ]; then
        echo "✓ Whisper dir exists: $WHISPER_DIR"
        ls -lh "$WHISPER_DIR"/transcript_*.csv 2>/dev/null || echo "  ✗ No transcript files"
    else
        echo "✗ Whisper dir missing: $WHISPER_DIR"
    fi

    # Check MFA input (Step 2 output)
    MFA_INPUT_INTERVIEWER="outputs/mfa_input/sub-03_interviewer/sub-03_run-0${RUN}_interviewer"
    if [ -d "$MFA_INPUT_INTERVIEWER" ]; then
        echo "✓ MFA input (interviewer) exists"
    else
        echo "✗ MFA input (interviewer) missing: $MFA_INPUT_INTERVIEWER"
    fi

    # Check MFA transcripts (Step 3 output)
    MFA_DIR="outputs/features/$SUBJECT/run-0$RUN"
    if [ -f "$MFA_DIR/transcript_mfa_interviewer.csv" ]; then
        echo "✓ MFA transcript (interviewer) exists"
    else
        echo "✗ MFA transcript (interviewer) missing: $MFA_DIR/transcript_mfa_interviewer.csv"
    fi

    # Check sync params (Step 4 output)
    SYNC_FILE="outputs/sync/$SUBJECT/run-0$RUN/sync_params.json"
    if [ -f "$SYNC_FILE" ]; then
        echo "✓ Sync params exist"
    else
        echo "✗ Sync params missing: $SYNC_FILE"
    fi

    # Check TRF FIF (Step 5 output)
    TRF_FIF="outputs/trf/$SUBJECT/run-0$RUN/${SUBJECT}_run-0${RUN}_trf_raw.fif"
    if [ -f "$TRF_FIF" ]; then
        echo "✓ TRF FIF exists"
    else
        echo "✗ TRF FIF missing: $TRF_FIF"
    fi

    echo ""
done

echo "=================================================================="
echo "AUDIO FILES CHECK"
echo "=================================================================="

# Check if audio files exist (these are on OneDrive)
echo "Checking audio file paths from config..."
for RUN in 1 2 3 4 5; do
    BLOCK=$RUN
    echo "Block $BLOCK:"
    echo "  Should be: console_mic_B${BLOCK}.wav"
    echo "  Should be: subject_mic_B${BLOCK}.wav"
done
