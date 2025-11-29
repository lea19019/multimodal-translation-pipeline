#!/bin/bash

# Stop all Multimodal Translation Services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directory
SERVICES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="${SERVICES_DIR}/.service_pids"
LOGS_DIR="${SERVICES_DIR}/_logs"
ARCHIVE_DIR="${LOGS_DIR}/archive"

# Create archive directory if it doesn't exist
mkdir -p "${ARCHIVE_DIR}"

# Get timestamp for archive
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Stopping Multimodal Translation Services${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to archive log file
archive_log() {
    local service_name=$1
    local log_file="${LOGS_DIR}/${service_name}.log"
    
    if [ -f "${log_file}" ] && [ -s "${log_file}" ]; then
        local archive_file="${ARCHIVE_DIR}/${service_name}_${TIMESTAMP}.log"
        cp "${log_file}" "${archive_file}"
        print_info "Archived ${service_name}.log → ${archive_file}"
        # Clear the current log file
        > "${log_file}"
    fi
}

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if PID file exists
if [ ! -f "${PIDFILE}" ]; then
    print_warning "No PID file found. Services may not be running or were started manually."
    print_info "Attempting to find and stop services by port..."
    
    # Try to find processes by port
    for port in 8075 8076 8077 8078 8079; do
        pid=$(lsof -ti:${port} 2>/dev/null || echo "")
        if [ -n "${pid}" ]; then
            print_info "Found process on port ${port} (PID: ${pid})"
            kill ${pid} 2>/dev/null && print_success "Stopped process on port ${port}" || print_warning "Could not stop process on port ${port}"
        fi
    done
    exit 0
fi

# Stop services from PID file
print_info "Archiving log files..."
echo ""

# Archive logs before stopping
archive_log "api_gateway"
archive_log "asr"
archive_log "nmt"
archive_log "tts"
archive_log "evaluation_api"

echo ""
print_info "Stopping services..."
echo ""

while IFS=: read -r service pid; do
    if ps -p ${pid} > /dev/null 2>&1; then
        print_info "Stopping ${service} (PID: ${pid})..."
        kill ${pid} 2>/dev/null
        
        # Wait for process to terminate
        count=0
        while ps -p ${pid} > /dev/null 2>&1 && [ ${count} -lt 10 ]; do
            sleep 0.5
            count=$((count + 1))
        done
        
        # Force kill if still running
        if ps -p ${pid} > /dev/null 2>&1; then
            print_warning "${service} did not stop gracefully, forcing..."
            kill -9 ${pid} 2>/dev/null
        fi
        
        print_success "${service} stopped"
    else
        print_warning "${service} (PID: ${pid}) was not running"
    fi
done < "${PIDFILE}"

# Clean up PID file
rm -f "${PIDFILE}"

echo ""
echo -e "${BLUE}================================================${NC}"
print_success "All services stopped"
echo -e "${BLUE}================================================${NC}"
