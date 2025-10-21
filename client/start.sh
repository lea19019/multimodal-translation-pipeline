#!/bin/bash

# Quick Start Script for Multimodal Translation Client
# This script helps set up and run the client application

set -e

echo "=================================="
echo "Multimodal Translation Client"
echo "Quick Start Setup"
echo "=================================="
echo ""

# Check Node.js installation
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    echo "Please install Node.js 18+ from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18 or higher is required"
    echo "Current version: $(node -v)"
    exit 1
fi

echo "✅ Node.js $(node -v) detected"
echo ""

# Check if in correct directory
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found"
    echo "Please run this script from the client directory"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
if command -v npm &> /dev/null; then
    npm install
else
    echo "❌ npm not found"
    exit 1
fi

echo ""
echo "✅ Dependencies installed successfully"
echo ""

# Create .env.local if it doesn't exist
if [ ! -f ".env.local" ]; then
    echo "📝 Creating .env.local from template..."
    cp .env.example .env.local
    echo "✅ Created .env.local"
    echo ""
fi

# Check if backend services are running
echo "🔍 Checking backend services..."
echo ""

check_service() {
    local service_name=$1
    local service_url=$2
    
    if curl -s -f -o /dev/null "$service_url/health"; then
        echo "✅ $service_name is running"
        return 0
    else
        echo "❌ $service_name is not running"
        return 1
    fi
}

SERVICES_OK=true

check_service "API Gateway" "http://localhost:8000" || SERVICES_OK=false
check_service "ASR Service" "http://localhost:8001" || SERVICES_OK=false
check_service "NMT Service" "http://localhost:8002" || SERVICES_OK=false
check_service "TTS Service" "http://localhost:8003" || SERVICES_OK=false

echo ""

if [ "$SERVICES_OK" = false ]; then
    echo "⚠️  Some backend services are not running"
    echo "Please start the services first:"
    echo "  cd ../services"
    echo "  bash start_all_services.sh"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "=================================="
echo "🚀 Starting Development Server..."
echo "=================================="
echo ""
echo "The application will be available at:"
echo "  http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the development server
npm run dev
