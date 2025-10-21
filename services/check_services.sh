#!/bin/bash

# Check the status of all Multimodal Translation Services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directory
SERVICES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="${SERVICES_DIR}/.service_pids"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Multimodal Translation Services Status${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[RUNNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[NOT RUNNING]${NC} $1"
}

# Function to check service health endpoint
check_health() {
    local service_name=$1
    local port=$2
    
    local response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${port}/health 2>/dev/null)
    
    if [ "${response}" = "200" ]; then
        return 0
    else
        return 1
    fi
}

# Check services by port
echo "Service Status:"
echo ""

declare -A services=(
    ["API Gateway"]=8075
    ["ASR Service"]=8076
    ["NMT Service"]=8077
    ["TTS Service"]=8078
)

all_running=true

for service in "API Gateway" "ASR Service" "NMT Service" "TTS Service"; do
    port=${services[$service]}
    pid=$(lsof -ti:${port} 2>/dev/null || echo "")
    
    if [ -n "${pid}" ]; then
        if check_health "${service}" ${port}; then
            print_success "${service} (port ${port}, PID: ${pid}) - Health check: OK"
        else
            echo -e "${YELLOW}[RUNNING]${NC} ${service} (port ${port}, PID: ${pid}) - Health check: FAILED"
        fi
    else
        print_error "${service} (port ${port})"
        all_running=false
    fi
done

echo ""

# Check PID file
if [ -f "${PIDFILE}" ]; then
    echo "Tracked PIDs:"
    while IFS=: read -r service pid; do
        if ps -p ${pid} > /dev/null 2>&1; then
            echo "  ✓ ${service}: ${pid}"
        else
            echo "  ✗ ${service}: ${pid} (not running)"
        fi
    done < "${PIDFILE}"
    echo ""
fi

# Summary
echo -e "${BLUE}================================================${NC}"
if [ "${all_running}" = true ]; then
    echo -e "${GREEN}All services are running${NC}"
else
    echo -e "${YELLOW}Some services are not running${NC}"
    echo "Run: bash ${SERVICES_DIR}/start_all_services.sh"
fi
echo -e "${BLUE}================================================${NC}"
