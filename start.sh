#!/bin/bash

# Multimodal Translation Pipeline - Startup Script
# This script starts both the Python Model Manager and the Node.js API Gateway

echo "🚀 Starting Multimodal Translation Pipeline"
echo "==========================================="

# Check if we're in the right directory
if [ ! -d "model-manager" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Expected directories: model-manager/, frontend/"
    exit 1
fi

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $1 is already in use"
        return 1
    fi
    return 0
}

# Function to install Python dependencies
setup_python() {
    echo "📦 Setting up Python Model Manager..."
    cd model-manager
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "   Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo "   Installing Python dependencies..."
    pip install -r requirements.txt
    
    cd ..
    echo "✅ Python Model Manager setup complete"
}

# Function to install Node.js dependencies
setup_node() {
    echo "📦 Setting up Node.js API Gateway..."
    cd frontend
    
    # Install dependencies
    echo "   Installing Node.js dependencies..."
    npm install
    
    cd ..
    echo "✅ Node.js API Gateway setup complete"
}

# Function to start the Model Manager
start_model_manager() {
    echo "🐍 Starting Python Model Manager on port 8000..."
    cd model-manager
    source venv/bin/activate
    python run.py &
    MODEL_MANAGER_PID=$!
    cd ..
    
    # Wait for Model Manager to start
    echo "   Waiting for Model Manager to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            echo "✅ Model Manager is ready"
            return 0
        fi
        sleep 1
    done
    
    echo "❌ Model Manager failed to start"
    return 1
}

# Function to start the API Gateway
start_api_gateway() {
    echo "🌐 Starting Node.js API Gateway on port 3001..."
    cd frontend
    npm run dev:server &
    API_GATEWAY_PID=$!
    cd ..
    
    # Wait for API Gateway to start
    echo "   Waiting for API Gateway to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:3001/api/health >/dev/null 2>&1; then
            echo "✅ API Gateway is ready"
            return 0
        fi
        sleep 1
    done
    
    echo "❌ API Gateway failed to start"
    return 1
}

# Function to start the frontend
start_frontend() {
    echo "⚛️  Starting React Frontend on port 5173..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    
    # Wait for frontend to start
    echo "   Waiting for Frontend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:5173 >/dev/null 2>&1; then
            echo "✅ Frontend is ready"
            return 0
        fi
        sleep 1
    done
    
    echo "❌ Frontend failed to start"
    return 1
}

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    if [ ! -z "$MODEL_MANAGER_PID" ]; then
        kill $MODEL_MANAGER_PID 2>/dev/null
        echo "   Stopped Model Manager"
    fi
    
    if [ ! -z "$API_GATEWAY_PID" ]; then
        kill $API_GATEWAY_PID 2>/dev/null
        echo "   Stopped API Gateway"
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "   Stopped Frontend"
    fi
    
    # Kill any remaining processes on our ports
    pkill -f "python run.py" 2>/dev/null
    pkill -f "node.*3001" 2>/dev/null
    pkill -f "vite.*5173" 2>/dev/null
    
    echo "✅ Cleanup complete"
    exit 0
}

# Setup signal handlers
trap cleanup SIGINT SIGTERM

# Check ports
echo "🔍 Checking ports..."
check_port 8000 || (echo "   Please stop the service on port 8000 and try again"; exit 1)
check_port 3001 || (echo "   Please stop the service on port 3001 and try again"; exit 1)
check_port 5173 || (echo "   Please stop the service on port 5173 and try again"; exit 1)

# Setup dependencies
setup_python
setup_node

# Start services
echo ""
echo "🚀 Starting services..."
start_model_manager || (echo "❌ Failed to start Model Manager"; exit 1)
start_api_gateway || (echo "❌ Failed to start API Gateway"; cleanup; exit 1)
start_frontend || (echo "❌ Failed to start Frontend"; cleanup; exit 1)

echo ""
echo "🎉 All services are running!"
echo "==========================================="
echo "📋 Service URLs:"
echo "   🐍 Model Manager:    http://localhost:8000"
echo "   🌐 API Gateway:      http://localhost:3001"
echo "   ⚛️  Frontend:         http://localhost:5173"
echo ""
echo "📖 API Documentation:"
echo "   🐍 Model Manager:    http://localhost:8000/docs"
echo "   🌐 API Gateway:      http://localhost:3001/api/health"
echo ""
echo "🧪 Test the integration:"
echo "   cd model-manager && python test_api.py"
echo ""
echo "Press Ctrl+C to stop all services"
echo "==========================================="

# Keep the script running
while true; do
    sleep 1
done