# Model Manager Integration Documentation

## Overview

The Model Manager is a Python-based service responsible for loading, managing, and executing machine learning models for the multimodal translation pipeline. The API Gateway communicates with the Model Manager through a well-defined interface to orchestrate translation workflows.

---

## Architecture Overview

```
┌─────────────────┐    HTTP/gRPC     ┌─────────────────┐
│   API Gateway   │ ◄──────────────► │  Model Manager  │
│   (Node.js)     │                  │   (Python)      │
└─────────────────┘                  └─────────────────┘
        │                                     │
        │                                     │
        ▼                                     ▼
┌─────────────────┐                  ┌─────────────────┐
│    Frontend     │                  │   ML Models     │
│  (React/TS)     │                  │ (PyTorch/HF)    │
└─────────────────┘                  └─────────────────┘
```

---

## Communication Protocol

### Option 1: HTTP REST API (Recommended)
- **Protocol**: HTTP/1.1 with JSON payloads
- **Port**: 8000 (configurable)
- **Timeout**: 30s for single requests, 300s for batch operations
- **Content-Type**: `application/json` for text, `multipart/form-data` for files

### Option 2: gRPC (High Performance)
- **Protocol**: gRPC with Protocol Buffers
- **Port**: 8001 (configurable)
- **Streaming**: Supported for real-time translation
- **Timeout**: Configurable per request type

---

## Model Manager Endpoints

### Service Health

#### `GET /health`
Check if the Model Manager is healthy and models are loaded.

**API Gateway Request:**
```bash
GET http://model-manager:8000/health
```

**Model Manager Response:**
```json
{
  "status": "healthy",
  "loadedModels": {
    "asr": ["whisper-base", "whisper-large"],
    "nmt": ["opus-mt", "mbart-large"],
    "tts": ["espeak-ng", "tacotron2"]
  },
  "gpuMemory": {
    "total": "24GB",
    "used": "8GB",
    "available": "16GB"
  },
  "timestamp": "2025-09-22T10:30:00.000Z"
}
```

### Model Loading

#### `POST /models/load`
Load specific models into memory.

**API Gateway Request:**
```json
{
  "models": [
    {
      "type": "asr",
      "name": "whisper-large-v3",
      "config": {
        "device": "cuda:0",
        "precision": "float16"
      }
    },
    {
      "type": "nmt", 
      "name": "mbart-large-50",
      "config": {
        "device": "cuda:1",
        "batch_size": 32
      }
    }
  ]
}
```

**Model Manager Response:**
```json
{
  "success": true,
  "loadedModels": [
    {
      "type": "asr",
      "name": "whisper-large-v3",
      "status": "loaded",
      "memoryUsage": "2.1GB",
      "loadTime": 15.2
    },
    {
      "type": "nmt",
      "name": "mbart-large-50", 
      "status": "loaded",
      "memoryUsage": "3.8GB",
      "loadTime": 22.7
    }
  ]
}
```

### Text Translation

#### `POST /translate/text`
Perform neural machine translation.

**API Gateway Request:**
```json
{
  "model": "mbart-large-50",
  "sourceLang": "en",
  "targetLang": "es",
  "text": "Hello, how are you today?",
  "options": {
    "beamSize": 5,
    "lengthPenalty": 1.0,
    "maxLength": 512,
    "returnLogProbs": true,
    "numReturnSequences": 1
  }
}
```

**Model Manager Response:**
```json
{
  "success": true,
  "translation": {
    "text": "Hola, ¿cómo estás hoy?",
    "confidence": 0.94,
    "logProb": -2.1,
    "alternatives": [
      {
        "text": "Hola, ¿cómo te encuentras hoy?",
        "confidence": 0.87,
        "logProb": -2.8
      }
    ]
  },
  "metadata": {
    "processingTime": 0.15,
    "tokensGenerated": 8,
    "device": "cuda:1"
  }
}
```

### Speech Recognition

#### `POST /transcribe`
Convert speech to text using ASR models.

**API Gateway Request:**
```bash
POST http://model-manager:8000/transcribe
Content-Type: multipart/form-data

model: whisper-large-v3
language: auto
task: transcribe
audio: (binary audio data)
options: {"return_timestamps": true, "word_level_timestamps": true}
```

**Model Manager Response:**
```json
{
  "success": true,
  "transcription": {
    "text": "Hello, how are you today?",
    "language": "en",
    "confidence": 0.96,
    "segments": [
      {
        "text": "Hello,",
        "start": 0.0,
        "end": 0.8,
        "confidence": 0.98,
        "words": [
          {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.99},
          {"word": ",", "start": 0.5, "end": 0.8, "confidence": 0.95}
        ]
      },
      {
        "text": "how are you today?",
        "start": 0.9,
        "end": 2.1,
        "confidence": 0.94,
        "words": [
          {"word": "how", "start": 0.9, "end": 1.1, "confidence": 0.96},
          {"word": "are", "start": 1.1, "end": 1.3, "confidence": 0.97},
          {"word": "you", "start": 1.3, "end": 1.5, "confidence": 0.98},
          {"word": "today", "start": 1.5, "end": 1.9, "confidence": 0.92},
          {"word": "?", "start": 1.9, "end": 2.1, "confidence": 0.88}
        ]
      }
    ]
  },
  "metadata": {
    "processingTime": 0.32,
    "audioLength": 2.1,
    "device": "cuda:0",
    "modelVersion": "large-v3"
  }
}
```

### Text-to-Speech

#### `POST /synthesize`
Generate speech from text using TTS models.

**API Gateway Request:**
```json
{
  "model": "tacotron2",
  "text": "Hola, ¿cómo estás hoy?",
  "language": "es",
  "voice": "female",
  "options": {
    "samplingRate": 22050,
    "vocoder": "hifigan",
    "speed": 1.0,
    "pitch": 0.0,
    "emotion": "neutral"
  }
}
```

**Model Manager Response:**
```json
{
  "success": true,
  "synthesis": {
    "audioBase64": "UklGRnoGAABXQVZFZm10IBAAAAABAAEA...",
    "format": "wav",
    "samplingRate": 22050,
    "duration": 2.5,
    "channels": 1
  },
  "metadata": {
    "processingTime": 0.8,
    "phonemes": "ˈo.la ˈko.mo es.ˈtas ˈoj",
    "device": "cuda:1",
    "modelVersion": "v1.0"
  }
}
```

### Batch Processing

#### `POST /batch/process`
Process multiple requests in a single batch.

**API Gateway Request:**
```json
{
  "batchId": "batch_789",
  "requests": [
    {
      "id": "req_001",
      "type": "translate",
      "model": "mbart-large-50",
      "sourceLang": "en",
      "targetLang": "es",
      "text": "Hello world"
    },
    {
      "id": "req_002",
      "type": "transcribe",
      "model": "whisper-base",
      "audioBase64": "UklGRnoGAABXQVZFZm10IBAAAAABAAEA..."
    }
  ],
  "options": {
    "parallel": true,
    "maxWorkers": 4
  }
}
```

**Model Manager Response:**
```json
{
  "success": true,
  "batchId": "batch_789",
  "results": [
    {
      "id": "req_001",
      "success": true,
      "result": {
        "translation": {
          "text": "Hola mundo",
          "confidence": 0.97
        }
      },
      "processingTime": 0.12
    },
    {
      "id": "req_002", 
      "success": true,
      "result": {
        "transcription": {
          "text": "Good morning",
          "confidence": 0.89
        }
      },
      "processingTime": 0.28
    }
  ],
  "summary": {
    "totalRequests": 2,
    "successfulRequests": 2,
    "failedRequests": 0,
    "totalProcessingTime": 0.4,
    "parallelWorkers": 2
  }
}
```

---

## Error Handling

### Model Manager Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "MODEL_NOT_LOADED",
    "message": "Model 'whisper-large-v3' is not currently loaded",
    "type": "ModelError",
    "details": {
      "availableModels": ["whisper-base", "whisper-small"],
      "suggestion": "Load the model first using /models/load endpoint"
    }
  },
  "timestamp": "2025-09-22T10:30:00.000Z"
}
```

### Common Error Codes
| Code | Description | Recovery |
|------|-------------|----------|
| `MODEL_NOT_LOADED` | Requested model not in memory | Load model first |
| `INSUFFICIENT_MEMORY` | Not enough GPU memory | Unload unused models |
| `INVALID_AUDIO_FORMAT` | Audio format not supported | Convert to supported format |
| `TEXT_TOO_LONG` | Input text exceeds max length | Split into smaller chunks |
| `LANGUAGE_NOT_SUPPORTED` | Language not supported by model | Use supported language |
| `PROCESSING_TIMEOUT` | Request exceeded timeout | Retry with simpler options |
| `DEVICE_ERROR` | GPU/CUDA error | Check hardware status |

---

## API Gateway Integration Layer

### Connection Management
```javascript
// model-manager-client.js
class ModelManagerClient {
  constructor(baseUrl = 'http://model-manager:8000') {
    this.baseUrl = baseUrl;
    this.timeout = 30000;
  }

  async translate(request) {
    const response = await fetch(`${this.baseUrl}/translate/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      timeout: this.timeout
    });
    
    if (!response.ok) {
      throw new ModelManagerError(await response.json());
    }
    
    return response.json();
  }

  async transcribe(audioBuffer, options) {
    const formData = new FormData();
    formData.append('audio', audioBuffer);
    formData.append('model', options.model);
    formData.append('options', JSON.stringify(options));

    const response = await fetch(`${this.baseUrl}/transcribe`, {
      method: 'POST',
      body: formData,
      timeout: this.timeout
    });

    return response.json();
  }
}
```

### Request Translation Layer
```javascript
// translation-service.js
class TranslationService {
  constructor(modelManagerClient) {
    this.modelManager = modelManagerClient;
  }

  async processTranslationRequest(apiRequest) {
    // Convert API Gateway request to Model Manager format
    const modelRequest = this.convertToModelManagerRequest(apiRequest);
    
    try {
      switch (apiRequest.type) {
        case 'text-to-text':
          return await this.handleTextTranslation(modelRequest);
        case 'speech-to-text':
          return await this.handleSpeechTranscription(modelRequest);
        case 'text-to-speech':
          return await this.handleTextSynthesis(modelRequest);
        case 'speech-to-speech':
          return await this.handleSpeechToSpeech(modelRequest);
      }
    } catch (error) {
      return this.handleModelManagerError(error);
    }
  }

  convertToModelManagerRequest(apiRequest) {
    // Transform API Gateway request format to Model Manager format
    const pipeline = this.getPipelineConfig(apiRequest.pipelineId);
    
    return {
      model: this.selectModel(apiRequest.type, pipeline),
      sourceLang: apiRequest.sourceLang,
      targetLang: apiRequest.targetLang,
      input: apiRequest.input,
      options: this.mergeOptions(pipeline.defaultOptions, apiRequest.options)
    };
  }
}
```

---

## Model Manager Configuration

### Docker Compose Integration
```yaml
# docker-compose.yml
services:
  api-gateway:
    build: ./frontend
    ports:
      - "3003:3003"
    environment:
      - MODEL_MANAGER_URL=http://model-manager:8000
    depends_on:
      - model-manager

  model-manager:
    build: ./model-manager
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
      - MODEL_CACHE_DIR=/models
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
```

### Environment Variables
```bash
# Model Manager Configuration
MODEL_MANAGER_URL=http://localhost:8000
MODEL_MANAGER_TIMEOUT=30000
MODEL_MANAGER_RETRY_ATTEMPTS=3
MODEL_MANAGER_RETRY_DELAY=1000

# Model Configuration
DEFAULT_ASR_MODEL=whisper-base
DEFAULT_NMT_MODEL=opus-mt
DEFAULT_TTS_MODEL=espeak-ng
MODEL_CACHE_SIZE=8GB
GPU_MEMORY_FRACTION=0.8

# Performance Settings
MAX_BATCH_SIZE=32
PARALLEL_WORKERS=4
QUEUE_MAX_SIZE=100
```

---

## Monitoring and Logging

### Metrics Collection
The API Gateway should collect these metrics for Model Manager communication:

- **Request Latency**: Time taken for Model Manager responses
- **Throughput**: Requests per second to Model Manager
- **Error Rate**: Failed requests percentage
- **Model Usage**: Which models are used most frequently
- **Resource Utilization**: GPU memory, CPU usage from Model Manager

### Health Checks
```javascript
// health-monitor.js
class HealthMonitor {
  async checkModelManagerHealth() {
    try {
      const response = await fetch(`${MODEL_MANAGER_URL}/health`);
      const health = await response.json();
      
      return {
        status: health.status,
        loadedModels: health.loadedModels,
        responseTime: response.headers.get('X-Response-Time'),
        lastCheck: new Date().toISOString()
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        error: error.message,
        lastCheck: new Date().toISOString()
      };
    }
  }
}
```

### Logging Format
```json
{
  "timestamp": "2025-09-22T10:30:00.000Z",
  "level": "info",
  "service": "api-gateway",
  "component": "model-manager-client",
  "requestId": "req_12345",
  "action": "translate_request",
  "details": {
    "model": "mbart-large-50",
    "sourceLang": "en",
    "targetLang": "es",
    "inputLength": 25,
    "processingTime": 150,
    "success": true
  }
}
```

This documentation provides a comprehensive guide for implementing the communication layer between the API Gateway and Model Manager, ensuring robust, scalable, and maintainable integration.