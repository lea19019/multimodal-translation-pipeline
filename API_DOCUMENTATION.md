# API Gateway Documentation

## Overview

The API Gateway serves as the central interface for the Multimodal Translation Pipeline system. It provides REST endpoints for translation services, pipeline management, and communication with the model manager.

## Base URL

- **Development**: `http://localhost:3003`
- **Production**: `https://your-domain.com/api`

---

## API Endpoints

### Health Check

#### `GET /api/health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-09-22T10:30:00.000Z",
  "version": "1.0.0"
}
```

---

### Pipeline Management

#### `GET /api/pipelines`

Get all available translation pipelines.

**Response:**
```json
[
  {
    "id": "baseline",
    "name": "Baseline Pipeline",
    "description": "Basic translation pipeline for general use",
    "asrModel": "whisper-base",
    "nmtModel": "opus-mt",
    "ttsModel": "espeak-ng",
    "languages": {
      "supported": ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "ar"],
      "pairs": [
        {"source": "en", "target": "es"},
        {"source": "es", "target": "en"}
      ]
    },
    "capabilities": ["text-to-text", "text-to-speech", "speech-to-text", "speech-to-speech"],
    "performance": {
      "latency": "low",
      "accuracy": "medium",
      "resourceUsage": "low"
    }
  }
]
```

#### `GET /api/pipelines/:id`

Get specific pipeline details.

**Parameters:**
- `id` (string): Pipeline identifier

**Response:**
```json
{
  "id": "advanced",
  "name": "Advanced Pipeline",
  "description": "High-accuracy pipeline with latest models",
  "models": {
    "asr": {
      "name": "whisper-large",
      "version": "v3",
      "config": {
        "language": "auto",
        "task": "transcribe"
      }
    },
    "nmt": {
      "name": "mbart-large",
      "version": "50",
      "config": {
        "beam_size": 5,
        "length_penalty": 1.0
      }
    },
    "tts": {
      "name": "tacotron2",
      "version": "v1",
      "config": {
        "sampling_rate": 22050,
        "vocoder": "hifigan"
      }
    }
  },
  "status": "active",
  "lastUpdated": "2025-09-22T09:00:00.000Z"
}
```

---

### Translation Services

#### `POST /api/translate`

Perform translation using specified pipeline.

**Request Headers:**
```
Content-Type: application/json
# OR
Content-Type: multipart/form-data (for file uploads)
```

**Request Body (JSON):**
```json
{
  "type": "text-to-text",
  "mode": "free",
  "pipelineId": "baseline",
  "sourceLang": "en",
  "targetLang": "es",
  "input": {
    "text": "Hello, how are you today?"
  },
  "reference": "Hola, ¿cómo estás hoy?",
  "options": {
    "returnConfidence": true,
    "includeAlternatives": false
  }
}
```

**Request Body (Multipart Form Data):**
```
type: text-to-speech
mode: evaluation
pipelineId: advanced
sourceLang: en
targetLang: es
file: (audio file)
reference: Expected output text
```

**Translation Types:**
- `text-to-text`: Text translation
- `text-to-speech`: Text to audio synthesis
- `speech-to-text`: Audio transcription
- `speech-to-speech`: Audio translation with speech output

**Translation Modes:**
- `free`: Basic translation without evaluation
- `evaluation`: Translation with quality metrics

**Response (Text-to-Text):**
```json
{
  "success": true,
  "result": {
    "translatedText": "Hola, ¿cómo estás hoy?",
    "confidence": 0.94,
    "alternatives": [
      "Hola, ¿cómo te encuentras hoy?",
      "Hola, ¿qué tal estás hoy?"
    ],
    "metadata": {
      "processingTime": 150,
      "modelUsed": "mbart-large-50",
      "tokenCount": 8
    }
  },
  "evaluation": {
    "bleuScore": 0.89,
    "cometScore": 0.91,
    "metrics": {
      "precision": 0.92,
      "recall": 0.87,
      "f1Score": 0.89
    }
  },
  "timestamp": "2025-09-22T10:30:00.000Z"
}
```

**Response (Text-to-Speech):**
```json
{
  "success": true,
  "result": {
    "audioUrl": "/api/audio/temp_12345.wav",
    "audioFormat": "wav",
    "duration": 2.5,
    "metadata": {
      "processingTime": 800,
      "modelUsed": "tacotron2",
      "samplingRate": 22050
    }
  },
  "evaluation": {
    "mosScore": 4.2,
    "naturalness": 0.85,
    "intelligibility": 0.92
  },
  "timestamp": "2025-09-22T10:30:00.000Z"
}
```

**Response (Speech-to-Text):**
```json
{
  "success": true,
  "result": {
    "transcribedText": "Hello, how are you today?",
    "confidence": 0.96,
    "segments": [
      {
        "text": "Hello,",
        "start": 0.0,
        "end": 0.8,
        "confidence": 0.98
      },
      {
        "text": "how are you today?",
        "start": 0.9,
        "end": 2.1,
        "confidence": 0.94
      }
    ],
    "metadata": {
      "processingTime": 320,
      "modelUsed": "whisper-large-v3",
      "audioLength": 2.1
    }
  },
  "evaluation": {
    "wer": 0.0,
    "cer": 0.0,
    "accuracy": 1.0
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_PIPELINE",
    "message": "Pipeline 'invalid-id' not found",
    "details": {
      "availablePipelines": ["baseline", "advanced", "experimental"]
    }
  },
  "timestamp": "2025-09-22T10:30:00.000Z"
}
```

#### `POST /api/translate/batch`

Perform batch translation operations.

**Request Body:**
```json
{
  "pipelineId": "baseline",
  "mode": "evaluation",
  "requests": [
    {
      "id": "req_001",
      "type": "text-to-text",
      "sourceLang": "en",
      "targetLang": "es",
      "input": {
        "text": "Hello world"
      },
      "reference": "Hola mundo"
    },
    {
      "id": "req_002",
      "type": "text-to-text",
      "sourceLang": "en",
      "targetLang": "fr",
      "input": {
        "text": "Good morning"
      },
      "reference": "Bonjour"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "batchId": "batch_789",
  "results": [
    {
      "id": "req_001",
      "success": true,
      "result": {
        "translatedText": "Hola mundo",
        "confidence": 0.97
      },
      "evaluation": {
        "bleuScore": 1.0
      }
    },
    {
      "id": "req_002",
      "success": true,
      "result": {
        "translatedText": "Buenos días",
        "confidence": 0.89
      },
      "evaluation": {
        "bleuScore": 0.65
      }
    }
  ],
  "summary": {
    "totalRequests": 2,
    "successfulRequests": 2,
    "failedRequests": 0,
    "averageBleuScore": 0.825,
    "totalProcessingTime": 450
  }
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Malformed request body or missing required fields |
| `UNSUPPORTED_TYPE` | Translation type not supported by pipeline |
| `INVALID_PIPELINE` | Pipeline ID not found |
| `LANGUAGE_NOT_SUPPORTED` | Language pair not supported by pipeline |
| `FILE_TOO_LARGE` | Uploaded file exceeds size limit |
| `INVALID_FILE_FORMAT` | File format not supported |
| `MODEL_UNAVAILABLE` | Required model is not available or loaded |
| `QUOTA_EXCEEDED` | API rate limit or quota exceeded |
| `INTERNAL_ERROR` | Server-side processing error |

---

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Free tier**: 100 requests per hour
- **Premium tier**: 1000 requests per hour
- **Batch processing**: 10 batch requests per hour

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

---

## File Upload Specifications

### Supported Audio Formats
- **WAV**: Uncompressed, 16-bit, 16kHz-48kHz
- **MP3**: CBR/VBR, 16kHz-48kHz
- **M4A**: AAC encoding
- **FLAC**: Lossless compression

### File Size Limits
- **Audio files**: Maximum 100MB
- **Text files**: Maximum 10MB
- **Batch JSON**: Maximum 5MB

### Audio Requirements
- **Sample rate**: 16kHz minimum (22kHz recommended)
- **Channels**: Mono preferred, stereo supported
- **Duration**: Maximum 10 minutes per file
- **Bit depth**: 16-bit minimum

---

## Authentication

### API Key Authentication
```
Authorization: Bearer YOUR_API_KEY
```

### Request Signing (Premium)
For enhanced security, requests can be signed using HMAC-SHA256:
```
X-Signature: sha256=calculated_signature
X-Timestamp: 1640995200
```

---

## Webhook Integration

### Batch Processing Webhooks

For long-running batch operations, configure webhooks to receive completion notifications:

**Webhook Payload:**
```json
{
  "event": "batch.completed",
  "batchId": "batch_789",
  "status": "completed",
  "summary": {
    "totalRequests": 100,
    "successfulRequests": 98,
    "failedRequests": 2,
    "processingTime": 45000
  },
  "downloadUrl": "https://api.example.com/batches/batch_789/download",
  "timestamp": "2025-09-22T10:45:00.000Z"
}
```

---

## SDK Examples

### Python
```python
import requests

# Text translation
response = requests.post('http://localhost:3003/api/translate', 
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={
        'type': 'text-to-text',
        'pipelineId': 'baseline',
        'sourceLang': 'en',
        'targetLang': 'es',
        'input': {'text': 'Hello world'}
    }
)

result = response.json()
print(result['result']['translatedText'])
```

### JavaScript
```javascript
const response = await fetch('http://localhost:3003/api/translate', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    type: 'text-to-text',
    pipelineId: 'baseline',
    sourceLang: 'en',
    targetLang: 'es',
    input: { text: 'Hello world' }
  })
});

const result = await response.json();
console.log(result.result.translatedText);
```

### cURL
```bash
curl -X POST http://localhost:3003/api/translate \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "text-to-text",
    "pipelineId": "baseline",
    "sourceLang": "en",
    "targetLang": "es",
    "input": {"text": "Hello world"}
  }'
```