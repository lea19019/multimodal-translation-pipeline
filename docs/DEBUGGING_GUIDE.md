# Debugging Guide - Multimodal Translation Pipeline

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Common Issues](#common-issues)
3. [Component-Specific Debugging](#component-specific-debugging)
4. [Log Analysis](#log-analysis)
5. [Testing Tools](#testing-tools)
6. [Performance Monitoring](#performance-monitoring)

## 🔍 System Overview

The system consists of three interconnected components:

```
Frontend (React)     API Gateway (Node.js)     Model Manager (Python)
Port: 3000          Port: 3003                Port: 8000
├─ UI Components    ├─ Express Server         ├─ FastAPI App
├─ API Calls        ├─ Request Routing        ├─ ML Model Simulation
└─ State Management └─ Model Manager Client   └─ Dummy Responses
```

## 🚨 Common Issues

### 1. Port Conflicts
**Symptoms**: "EADDRINUSE" errors when starting services
**Solution**:
```bash
# Check what's using ports
lsof -i :3000  # Frontend
lsof -i :3003  # API Gateway  
lsof -i :8000  # Model Manager

# Kill processes
pkill -f "npm run dev"
pkill -f "node.*3003"
pkill -f "python run.py"
```

### 2. Model Manager Connection Issues
**Symptoms**: 404 errors, "Failed after 3 attempts" messages
**Debug Steps**:
```bash
# 1. Check if Model Manager is running
curl http://localhost:8000/health

# 2. Check API Gateway Model Manager client config
grep -n "MODEL_MANAGER_URL" frontend/src/server/model-manager-client.ts

# 3. Test direct Model Manager endpoints
curl http://localhost:8000/pipelines
```

### 3. Frontend Display Issues
**Symptoms**: "Binary audio data received" for text translations
**Debug**:
- Check browser developer console for errors
- Verify API response structure matches frontend expectations
- Check `TranslationInterface.tsx` output rendering logic

### 4. CORS Issues
**Symptoms**: Cross-origin request blocked errors
**Solution**:
- Ensure API Gateway has CORS enabled
- Check Vite proxy configuration in `vite.config.ts`

## 🧩 Component-Specific Debugging

### Frontend (React + Vite)

**Log Locations**:
- Browser Developer Console (F12)
- Terminal running `npm run dev`

**Debug Commands**:
```bash
# Start with verbose logging
npm run dev -- --debug

# Check build issues
npm run build

# Check TypeScript errors
npx tsc --noEmit
```

**Key Files to Check**:
- `frontend/src/client/components/TranslationInterface.tsx` - Main UI logic
- `frontend/vite.config.ts` - Proxy configuration
- `frontend/src/shared/types.ts` - Type definitions

**Common Issues**:
```javascript
// Check API call structure
console.log('Request:', requestData);
console.log('Response:', response);

// Verify state updates
console.log('Translation result:', result);
```

### API Gateway (Node.js + Express)

**Log Locations**:
- Terminal running `npm run dev:server`
- Console.log statements in server code

**Debug Commands**:
```bash
# Start with debug mode
DEBUG=* npm run dev:server

# Check TypeScript compilation
npx tsx check src/server/index.ts

# Test endpoints directly
curl -X POST http://localhost:3003/api/translate \
  -H "Content-Type: application/json" \
  -d '{"type": "text-to-text", "input": "test"}'
```

**Key Files to Check**:
- `frontend/src/server/index.ts` - Main server file
- `frontend/src/server/model-manager-client.ts` - Model Manager communication
- `frontend/src/server/translation-service.ts` - Translation logic

**Debug Code Examples**:
```typescript
// Add to translation endpoint
console.log('📥 Received translation request');
console.log('Body keys:', Object.keys(req.body));
console.log('Request data:', requestData);

// Add to Model Manager client
console.log('🔄 Making request to Model Manager:', url);
console.log('Request payload:', payload);
```

### Model Manager (Python + FastAPI)

**Log Locations**:
- Terminal running `python run.py`
- FastAPI automatic request logging
- Custom logging in endpoints

**Debug Commands**:
```bash
# Start with debug mode
cd model-manager
source venv/bin/activate
python run.py  # Check config.py for debug=True

# Test endpoints
python test_api.py

# Check specific endpoint
curl http://localhost:8000/translate/text \
  -H "Content-Type: application/json" \
  -d '{"model": "opus-mt", "sourceLang": "en", "targetLang": "es", "text": "hello"}'
```

**Key Files to Check**:
- `model-manager/main.py` - FastAPI app and endpoints
- `model-manager/models.py` - Data models and dummy responses
- `model-manager/config.py` - Configuration settings

**Debug Code Examples**:
```python
# Add to endpoints
import logging
logger = logging.getLogger(__name__)

@app.post("/translate/text")
async def translate_text(request: TranslationRequest):
    logger.info(f"Translation request: {request.model_dump()}")
    # ... rest of function
```

## 📊 Log Analysis

### API Gateway Logs
```bash
# Look for these patterns:
grep -E "(📥|🔄|❌|✅)" frontend_logs.txt

# Common log messages:
📥 Received translation request     # Request received
🔄 Processing text-to-text          # Processing started  
❌ Model Manager request failed     # Connection issue
✅ Translation completed            # Success
```

### Model Manager Logs
```bash
# FastAPI access logs show:
INFO:     127.0.0.1:56284 - "POST /translate/text HTTP/1.1" 200 OK

# Application logs show:
🚀 Model Manager starting on 0.0.0.0:8000
📋 Loaded models: {'asr': ['whisper-base'], 'nmt': ['opus-mt']}
```

### Frontend Console Logs
```javascript
// Look for:
"Translation result:"     // Successful response
"API Error:"             // Request failures
"TypeError:"             // Type mismatches
```

## 🔧 Testing Tools

### 1. API Testing Script
```bash
# Test all endpoints quickly
cd model-manager && python test_api.py
```

### 2. Manual API Testing
```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:3003/api/health

# Translation test
curl -X POST http://localhost:3003/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "type": "text-to-text",
    "sourceLang": "en", 
    "targetLang": "es",
    "input": "Hello world"
  }' | jq .
```

### 3. Component Testing
```bash
# TypeScript type checking
npx tsc --noEmit

# React component testing (if tests exist)
npm test

# Linting
npm run lint
```

## 📈 Performance Monitoring

### Response Time Monitoring
```bash
# Time API calls
time curl -X POST http://localhost:3003/api/translate \
  -H "Content-Type: application/json" \
  -d '{"type": "text-to-text", "input": "test"}'
```

### Memory Usage
```bash
# Check Node.js memory
ps aux | grep node

# Check Python memory  
ps aux | grep python

# Check port usage
netstat -tulpn | grep -E "(3000|3003|8000)"
```

### Error Rate Monitoring
```bash
# Count errors in logs
grep -c "ERROR\|Failed\|Exception" model-manager/logs/*.log
grep -c "❌" api-gateway/logs/*.log
```

## 🛠️ Debugging Workflows

### 1. Request Not Reaching Backend
```bash
# Step 1: Check frontend network tab
# Step 2: Verify API Gateway is running
curl http://localhost:3003/api/health

# Step 3: Check CORS and proxy settings
grep -A 10 "proxy" frontend/vite.config.ts
```

### 2. Backend Processing Issues
```bash
# Step 1: Check Model Manager directly
curl http://localhost:8000/health

# Step 2: Test API Gateway to Model Manager connection
grep -A 5 "ModelManagerClient" frontend/src/server/model-manager-client.ts

# Step 3: Check request transformation
grep -A 10 "processTranslation" frontend/src/server/translation-service.ts
```

### 3. Response Format Issues
```bash
# Step 1: Check actual API response
curl -s http://localhost:3003/api/translate ... | jq .

# Step 2: Compare with frontend expectations
grep -A 10 "result\." frontend/src/client/components/TranslationInterface.tsx

# Step 3: Verify type definitions
cat frontend/src/shared/types.ts
```

## 🚨 Emergency Recovery

### Complete System Reset
```bash
# Stop all processes
pkill -f "npm"
pkill -f "python.*run.py"
pkill -f "node"

# Clean and restart
cd /path/to/project
./start.sh
```

### Reset Python Environment
```bash
cd model-manager
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-simple.txt
```

### Reset Node Dependencies
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## 📞 Quick Reference

### Service Status Check
```bash
# All services health check
curl http://localhost:3000     # Frontend
curl http://localhost:3003/api/health  # API Gateway  
curl http://localhost:8000/health      # Model Manager
```

### Log Locations
- **Frontend**: Browser DevTools Console
- **API Gateway**: Terminal output from `npm run dev:server`
- **Model Manager**: Terminal output from `python run.py`

### Configuration Files
- **Frontend**: `frontend/vite.config.ts`, `frontend/src/shared/types.ts`
- **API Gateway**: `frontend/src/server/index.ts`
- **Model Manager**: `model-manager/config.py`, `model-manager/main.py`

### Key URLs
- Frontend: http://localhost:3000
- API Gateway: http://localhost:3003
- Model Manager: http://localhost:8000
- API Documentation: http://localhost:8000/docs