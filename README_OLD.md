# Multimodal Translation Pipeline - Documentation Index

## Overview

This repository contains a comprehensive multimodal translation pipeline system with a modern web interface and robust API gateway. The system supports text-to-text, text-to-speech, speech-to-text, and speech-to-speech translation workflows.

## Documentation Files

### 1. [Frontend README](frontend/README.md)
**Purpose**: Getting started guide and basic usage
- Installation and setup instructions
- Development and production build commands
- Basic feature overview
- Architecture summary

### 2. [API Documentation](API_DOCUMENTATION.md)
**Purpose**: Complete API reference for developers
- **Audience**: Frontend developers, integration partners, API consumers
- **Contents**:
  - All REST endpoints with request/response examples
  - Authentication and rate limiting
  - Error codes and handling
  - File upload specifications
  - SDK examples in Python, JavaScript, and cURL
  - Webhook integration for batch processing

### 3. [Model Manager Integration](MODEL_MANAGER_INTEGRATION.md)
**Purpose**: Backend integration specifications
- **Audience**: Backend developers, DevOps engineers, ML engineers
- **Contents**:
  - Communication protocol between API Gateway and Model Manager
  - HTTP/gRPC endpoint specifications
  - Model loading and management
  - Batch processing workflows
  - Error handling and recovery
  - Docker deployment configuration
  - Monitoring and logging strategies

## System Architecture

```
┌─────────────────┐    HTTP API     ┌─────────────────┐    HTTP/gRPC    ┌─────────────────┐
│   Frontend      │ ◄──────────────► │  API Gateway    │ ◄──────────────► │  Model Manager  │
│   (React/TS)    │                  │   (Node.js)     │                  │   (Python)      │
└─────────────────┘                  └─────────────────┘                  └─────────────────┘
        │                                     │                                     │
        ▼                                     ▼                                     ▼
   Web Interface                       REST Endpoints                        ML Models
   - Translation UI                    - /api/translate                     - Whisper (ASR)
   - Pipeline Config                   - /api/pipelines                     - mBART (NMT)  
   - Metrics Dashboard                 - /api/health                        - Tacotron2 (TTS)
   - Live Recording                    - File Upload                        - Custom Models
```

## Quick Reference

### For Frontend Developers
- **Start here**: [Frontend README](frontend/README.md)
- **API Integration**: [API Documentation](API_DOCUMENTATION.md)
- **Base URL**: `http://localhost:3003/api`
- **Key Endpoints**: `/translate`, `/pipelines`

### For Backend Developers  
- **Start here**: [Model Manager Integration](MODEL_MANAGER_INTEGRATION.md)
- **Protocol**: HTTP REST (recommended) or gRPC
- **Model Manager URL**: `http://localhost:8000`
- **Key Endpoints**: `/translate/text`, `/transcribe`, `/synthesize`

### For DevOps Engineers
- **Frontend Port**: 3000 (dev), 3003 (API)
- **Model Manager Port**: 8000 (HTTP), 8001 (gRPC) 
- **Dependencies**: Node.js 18+, Python 3.8+, CUDA (optional)
- **Docker**: See [Model Manager Integration](MODEL_MANAGER_INTEGRATION.md)

## Implementation Status

### ✅ Completed
- [x] Frontend React application with TypeScript
- [x] API Gateway with Express.js
- [x] Live audio recording with MediaRecorder API
- [x] File upload support (audio, text, batch JSON)
- [x] Pipeline configuration management
- [x] Mock translation endpoints
- [x] Dark-themed responsive UI
- [x] Comprehensive API documentation
- [x] Model Manager integration specifications

### 🔄 Ready for Implementation
- [ ] Python Model Manager service
- [ ] Actual ML model integration (Whisper, mBART, Tacotron2)
- [ ] Database for pipeline configurations
- [ ] User authentication and API keys
- [ ] Rate limiting and quotas
- [ ] Production deployment configuration

### 📋 Future Enhancements
- [ ] Real-time streaming translation
- [ ] Custom model fine-tuning interface
- [ ] Advanced analytics and reporting
- [ ] Multi-tenant support
- [ ] gRPC streaming for large files
- [ ] Kubernetes deployment manifests

## Getting Started

1. **Development Setup**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

2. **API Testing**
   ```bash
   curl -X POST http://localhost:3003/api/translate \
     -H "Content-Type: application/json" \
     -d '{"type":"text-to-text","input":{"text":"Hello"},"sourceLang":"en","targetLang":"es"}'
   ```

3. **Model Manager Integration**
   - Follow [Model Manager Integration](MODEL_MANAGER_INTEGRATION.md)
   - Implement Python service at `http://localhost:8000`
   - Update API Gateway to call real Model Manager

## Support and Contributing

### Issue Categories
- **Frontend Issues**: UI/UX, React components, TypeScript errors
- **API Issues**: Endpoint behavior, request/response format
- **Integration Issues**: Model Manager communication, Docker deployment
- **Documentation**: Improvements to guides and examples

### Development Workflow
1. Check existing documentation for guidance
2. Follow TypeScript strict mode for frontend
3. Use comprehensive error handling for API endpoints
4. Include tests for new Model Manager endpoints
5. Update relevant documentation files

---

**Last Updated**: September 22, 2025  
**Version**: 1.0.0  
**Contributors**: Development Team