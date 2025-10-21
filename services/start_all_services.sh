#!/bin/bash

# Multimodal Translation Services Launcher
# This script starts all microservices for the translation pipeline

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages (defined early for use during initialization)
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

# Base directory
SERVICES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SERVICES_DIR}/_logs"

# Load environment variables from .env file if it exists
if [ -f "${SERVICES_DIR}/.env" ]; then
    print_info "Loading environment variables from .env file..."
    export $(grep -v '^#' "${SERVICES_DIR}/.env" | xargs)
fi

# Default port configuration (can be overridden by .env file)
GATEWAY_PORT=${GATEWAY_PORT:-8075}
ASR_PORT=${ASR_PORT:-8076}
NMT_PORT=${NMT_PORT:-8077}
TTS_PORT=${TTS_PORT:-8078}

# Create _logs directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# PID file to track running services
PIDFILE="${SERVICES_DIR}/.service_pids"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Multimodal Translation Services Launcher${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to start a service
start_service() {
    local service_name=$1
    local service_dir=$2
    local service_file=$3
    local port=$4
    
    print_info "Starting ${service_name} on port ${port}..."
    
    cd "${SERVICES_DIR}/${service_dir}"
    
    # Check if service file exists
    if [ ! -f "${service_file}" ]; then
        print_error "Service file ${service_file} not found in ${service_dir}"
        return 1
    fi
    
    # Start service in background and redirect output to log file
    nohup uv run python "${service_file}" > "${LOG_DIR}/${service_name}.log" 2>&1 &
    local pid=$!
    
    # Save PID
    echo "${service_name}:${pid}" >> "${PIDFILE}"
    
    # Wait a moment and check if process is still running
    sleep 2
    if ps -p ${pid} > /dev/null 2>&1; then
        print_success "${service_name} started successfully (PID: ${pid})"
        return 0
    else
        print_error "${service_name} failed to start. Check ${LOG_DIR}/${service_name}.log"
        return 1
    fi
}

# Function to check if services are already running
check_existing_services() {
    if [ -f "${PIDFILE}" ]; then
        print_warning "Found existing PID file. Checking for running services..."
        
        local has_running=false
        local running_services=""
        
        while IFS=: read -r service pid; do
            if ps -p ${pid} > /dev/null 2>&1; then
                has_running=true
                running_services="${running_services}  - ${service} (PID: ${pid})\n"
            fi
        done < "${PIDFILE}"
        
        if [ "$has_running" = true ]; then
            echo -e "${YELLOW}Running services:${NC}"
            echo -e "${running_services}"
            echo -e "${YELLOW}Do you want to stop all services and restart? (y/n)${NC}"
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                print_info "Stopping all services..."
                bash "${SERVICES_DIR}/stop_all_services.sh"
                sleep 1
                rm -f "${PIDFILE}"
                return 0
            else
                print_error "Aborting. Please stop existing services first."
                print_info "Run: bash ${SERVICES_DIR}/stop_all_services.sh"
                exit 1
            fi
        fi
    fi
    # Clean up PID file if no services running
    rm -f "${PIDFILE}"
}

# Main execution
main() {
    # Check for existing services
    check_existing_services
    
    print_info "Starting all services..."
    echo ""
    
    # Start services in order: ASR, NMT, TTS, then API Gateway
    local failed=0
    
    # Start ASR Service
    if ! start_service "asr" "asr" "service.py" "${ASR_PORT}"; then
        failed=1
    fi
    echo ""
    
    # Start NMT Service
    if ! start_service "nmt" "nmt" "service.py" "${NMT_PORT}"; then
        failed=1
    fi
    echo ""
    
    # Start TTS Service
    if ! start_service "tts" "tts" "service.py" "${TTS_PORT}"; then
        failed=1
    fi
    echo ""
    
    # Start API Gateway
    if ! start_service "api_gateway" "api_gateway" "api.py" "${GATEWAY_PORT}"; then
        failed=1
    fi
    echo ""
    
    # Summary
    echo -e "${BLUE}================================================${NC}"
    if [ ${failed} -eq 0 ]; then
        print_success "All services started successfully!"
    else
        print_warning "Some services failed to start. Check logs in ${LOG_DIR}/"
    fi
    echo -e "${BLUE}================================================${NC}"
    echo ""
    
    print_info "Service endpoints:"
    echo "  - API Gateway:  http://localhost:${GATEWAY_PORT}"
    echo "  - ASR Service:  http://localhost:${ASR_PORT}"
    echo "  - NMT Service:  http://localhost:${NMT_PORT}"
    echo "  - TTS Service:  http://localhost:${TTS_PORT}"
    echo ""
    
    print_info "Logs are available in: ${LOG_DIR}/"
    print_info "To stop all services, run: bash ${SERVICES_DIR}/stop_all_services.sh"
    print_info "To view logs in real-time: tail -f ${LOG_DIR}/<service_name>.log"
    echo ""
    
    # Show PIDs
    print_info "Service PIDs:"
    if [ -f "${PIDFILE}" ]; then
        while IFS=: read -r service pid; do
            echo "  - ${service}: ${pid}"
        done < "${PIDFILE}"
    fi
}

# Run main function
main
