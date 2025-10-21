#!/bin/bash

# View logs for Multimodal Translation Services

# Colors for output
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directory
SERVICES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SERVICES_DIR}/_logs"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Multimodal Translation Services - Log Viewer${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if _logs directory exists
if [ ! -d "${LOG_DIR}" ]; then
    echo "No _logs directory found. Services may not have been started yet."
    exit 1
fi

# Check if any log files exist
if [ -z "$(ls -A ${LOG_DIR})" ]; then
    echo "No log files found in ${LOG_DIR}"
    exit 1
fi

# Show available logs
echo "Available log files:"
echo ""
ls -lh "${LOG_DIR}"/*.log 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# Menu for log selection
echo "Select an option:"
echo "  1) View API Gateway logs"
echo "  2) View ASR Service logs"
echo "  3) View NMT Service logs"
echo "  4) View TTS Service logs"
echo "  5) View all logs (combined)"
echo "  6) Follow API Gateway logs (live)"
echo "  7) Follow ASR Service logs (live)"
echo "  8) Follow NMT Service logs (live)"
echo "  9) Follow TTS Service logs (live)"
echo "  10) Follow all logs (live)"
echo "  q) Quit"
echo ""
read -p "Enter choice: " choice

case ${choice} in
    1)
        if [ -f "${LOG_DIR}/api_gateway.log" ]; then
            less +G "${LOG_DIR}/api_gateway.log"
        else
            echo "API Gateway log not found"
        fi
        ;;
    2)
        if [ -f "${LOG_DIR}/asr.log" ]; then
            less +G "${LOG_DIR}/asr.log"
        else
            echo "ASR Service log not found"
        fi
        ;;
    3)
        if [ -f "${LOG_DIR}/nmt.log" ]; then
            less +G "${LOG_DIR}/nmt.log"
        else
            echo "NMT Service log not found"
        fi
        ;;
    4)
        if [ -f "${LOG_DIR}/tts.log" ]; then
            less +G "${LOG_DIR}/tts.log"
        else
            echo "TTS Service log not found"
        fi
        ;;
    5)
        cat "${LOG_DIR}"/*.log 2>/dev/null | less +G
        ;;
    6)
        if [ -f "${LOG_DIR}/api_gateway.log" ]; then
            tail -f "${LOG_DIR}/api_gateway.log"
        else
            echo "API Gateway log not found"
        fi
        ;;
    7)
        if [ -f "${LOG_DIR}/asr.log" ]; then
            tail -f "${LOG_DIR}/asr.log"
        else
            echo "ASR Service log not found"
        fi
        ;;
    8)
        if [ -f "${LOG_DIR}/nmt.log" ]; then
            tail -f "${LOG_DIR}/nmt.log"
        else
            echo "NMT Service log not found"
        fi
        ;;
    9)
        if [ -f "${LOG_DIR}/tts.log" ]; then
            tail -f "${LOG_DIR}/tts.log"
        else
            echo "TTS Service log not found"
        fi
        ;;
    10)
        tail -f "${LOG_DIR}"/*.log
        ;;
    q|Q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
