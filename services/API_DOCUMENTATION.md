# Multimodal Translation API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [API Gateway](#api-gateway)
5. [Individual Services](#individual-services)
   - [ASR Service](#asr-service)
   - [NMT Service](#nmt-service)
   - [TTS Service](#tts-service)
6. [Client Implementation Guide](#client-implementation-guide)
7. [Error Handling](#error-handling)
8. [Language Codes](#language-codes)
9. [Audio Format Specifications](#audio-format-specifications)

---

## Overview

The Multimodal Translation System is a microservices-based platform that provides:
- **Text-to-Text Translation**: Translate text between languages
- **Audio-to-Text Translation**: Transcribe and translate spoken audio
- **Text-to-Audio Translation**: Translate text and synthesize speech
- **Audio-to-Audio Translation**: Full pipeline for spoken language translation

### Key Features
- RESTful API design
- Microservices architecture with independent scaling
- Model selection support (multiple models per service)
- Base64 audio encoding for easy HTTP transport
- Comprehensive error handling and health checks

---

## Architecture

```
┌─────────────────────────────────────────┐
│         API Gateway (Port 8075)         │
│     Main Entry Point for All Requests   │
└───────────────┬─────────────────────────┘
                │
    ┌───────────┼───────────┬─────────────┐
    │           │           │             │
┌───▼────┐ ┌───▼────┐ ┌───▼────┐   (Direct Path)
│  ASR   │ │  NMT   │ │  TTS   │
│  8076  │ │  8077  │ │  8078  │
└────────┘ └────────┘ └────────┘
```

### Service Responsibilities

| Service | Port | Purpose | Technology |
|---------|------|---------|------------|
| API Gateway | 8075 | Request orchestration, pipeline coordination | FastAPI, httpx |
| ASR Service | 8076 | Automatic Speech Recognition | Whisper (transformers) |
| NMT Service | 8077 | Neural Machine Translation | NLLB (transformers) |
| TTS Service | 8078 | Text-to-Speech Synthesis | Coqui XTTS |

---

## Getting Started

### Base URLs

**Default (local deployment):**
```
API Gateway:  http://localhost:8075
ASR Service:  http://localhost:8076
NMT Service:  http://localhost:8077
TTS Service:  http://localhost:8078
```

### Environment Configuration

Services can be configured via environment variables. Create a `.env` file in the `services/` directory:

```bash
# Port Configuration (using 8075-8078 to avoid conflicts on shared systems)
GATEWAY_PORT=8075
ASR_PORT=8076
NMT_PORT=8077
TTS_PORT=8078

# Service URL Override (optional, for distributed deployments)
# ASR_SERVICE_URL=http://asr-server:8076
# NMT_SERVICE_URL=http://nmt-server:8077
# TTS_SERVICE_URL=http://tts-server:8078
```

### Quick Health Check

```bash
# Check all services via API Gateway
curl http://localhost:8075/health

# Check individual services
curl http://localhost:8076/health
curl http://localhost:8077/health
curl http://localhost:8078/health
```

---

## API Gateway

**Base URL:** `http://localhost:8075`

The API Gateway is the **recommended entry point** for all translation operations. It orchestrates the entire pipeline and handles service coordination.

### Endpoints

#### 1. Health Check
Check the status of the API Gateway and all downstream services.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "api_gateway",
  "downstream_services": {
    "asr": {
      "status": "healthy",
      "service": "asr",
      "device": "cpu"
    },
    "nmt": {
      "status": "healthy",
      "service": "nmt",
      "device": "cpu"
    },
    "tts": {
      "status": "healthy",
      "service": "tts",
      "device": "cpu"
    }
  }
}
```

**Example:**
```bash
curl http://localhost:8075/health
```

---

#### 2. Translate
Main translation endpoint that handles the complete translation pipeline.

**Endpoint:** `POST /translate`

**Request Body:**
```json
{
  "input": "string",              // Text or base64 encoded audio
  "input_type": "text | audio",   // Type of input
  "source_language": "string",    // Source language code (e.g., "en", "es", "fr")
  "target_language": "string",    // Target language code
  "output_type": "text | audio",  // Desired output type
  "asr_model": "string",          // Optional: ASR model name (default: "base")
  "nmt_model": "string",          // Optional: NMT model name (default: "base")
  "tts_model": "string"           // Optional: TTS model name (default: "base")
}
```

**Response:**
```json
{
  "output": "string",              // Translated text or base64 encoded audio
  "output_type": "text | audio",   // Type of output
  "source_language": "string",     // Source language code
  "target_language": "string"      // Target language code
}
```

**Pipeline Routing Logic:**
- `input_type="audio"` → Calls ASR Service first
- Always calls NMT Service for translation
- `output_type="audio"` → Calls TTS Service last
- `input_type="text"` and `output_type="text"` → Only NMT Service

---

### Use Cases and Examples

#### Use Case 1: Text-to-Text Translation
Translate text from English to Spanish.

**Request:**
```bash
curl -X POST http://localhost:8075/translate \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, how are you?",
    "input_type": "text",
    "source_language": "en",
    "target_language": "es",
    "output_type": "text",
    "nmt_model": "base"
  }'
```

**Response:**
```json
{
  "output": "Hola, ¿cómo estás?",
  "output_type": "text",
  "source_language": "en",
  "target_language": "es"
}
```

---

#### Use Case 2: Audio-to-Text Translation
Transcribe and translate spoken English to Spanish text.

**Request:**
```bash
# First, encode your audio file to base64
AUDIO_BASE64=$(base64 -w 0 audio_file.wav)

curl -X POST http://localhost:8075/translate \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": \"$AUDIO_BASE64\",
    \"input_type\": \"audio\",
    \"source_language\": \"en\",
    \"target_language\": \"es\",
    \"output_type\": \"text\",
    \"asr_model\": \"base\",
    \"nmt_model\": \"base\"
  }"
```

**Response:**
```json
{
  "output": "Hola, ¿cómo estás?",
  "output_type": "text",
  "source_language": "en",
  "target_language": "es"
}
```

---

#### Use Case 3: Text-to-Audio Translation
Translate English text to Spanish and synthesize speech.

**Request:**
```bash
curl -X POST http://localhost:8075/translate \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, how are you?",
    "input_type": "text",
    "source_language": "en",
    "target_language": "es",
    "output_type": "audio",
    "nmt_model": "base",
    "tts_model": "base"
  }'
```

**Response:**
```json
{
  "output": "<base64_encoded_audio>",
  "output_type": "audio",
  "source_language": "en",
  "target_language": "es"
}
```

**Decoding Audio Output:**
```bash
# Save the base64 audio to a file and decode
echo "<base64_encoded_audio>" | base64 -d > output_audio.raw
```

---

#### Use Case 4: Audio-to-Audio Translation
Full pipeline: transcribe English audio, translate to Spanish, and synthesize speech.

**Request:**
```bash
AUDIO_BASE64=$(base64 -w 0 input_audio.wav)

curl -X POST http://localhost:8075/translate \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": \"$AUDIO_BASE64\",
    \"input_type\": \"audio\",
    \"source_language\": \"en\",
    \"target_language\": \"es\",
    \"output_type\": \"audio\",
    \"asr_model\": \"base\",
    \"nmt_model\": \"base\",
    \"tts_model\": \"base\"
  }"
```

**Response:**
```json
{
  "output": "<base64_encoded_audio>",
  "output_type": "audio",
  "source_language": "en",
  "target_language": "es"
}
```

---

## Individual Services

You can also call individual services directly for specific operations.

### ASR Service

**Base URL:** `http://localhost:8076`

Automatic Speech Recognition service using OpenAI Whisper models.

#### Endpoints

##### 1. Health Check
**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "asr",
  "device": "cpu"
}
```

##### 2. List Models
**Endpoint:** `GET /models`

**Response:**
```json
{
  "models": ["base", "small", "medium"]
}
```

##### 3. Transcribe Audio
**Endpoint:** `POST /transcribe`

**Request:**
```json
{
  "audio": "string",              // Base64 encoded audio (float32, 16kHz)
  "source_language": "string",    // Optional: Language code (default: "en")
  "model_name": "string"          // Optional: Model name (default: "base")
}
```

**Response:**
```json
{
  "text": "string",               // Transcribed text
  "language": "string"            // Language code
}
```

**Example:**
```bash
AUDIO_BASE64=$(base64 -w 0 audio.wav)

curl -X POST http://localhost:8076/transcribe \
  -H "Content-Type: application/json" \
  -d "{
    \"audio\": \"$AUDIO_BASE64\",
    \"source_language\": \"en\",
    \"model_name\": \"base\"
  }"
```

**Audio Requirements:**
- Format: Raw PCM audio as numpy float32 array
- Sample Rate: 16kHz
- Channels: Mono
- Encoding: Base64

---

### NMT Service

**Base URL:** `http://localhost:8077`

Neural Machine Translation service using Meta's NLLB (No Language Left Behind) models.

#### Endpoints

##### 1. Health Check
**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "nmt",
  "device": "cpu"
}
```

##### 2. List Models
**Endpoint:** `GET /models`

**Response:**
```json
{
  "models": ["base", "large"]
}
```

##### 3. Translate Text
**Endpoint:** `POST /translate`

**Request:**
```json
{
  "text": "string",               // Text to translate
  "source_language": "string",    // Source language code
  "target_language": "string",    // Target language code
  "model_name": "string"          // Optional: Model name (default: "base")
}
```

**Response:**
```json
{
  "translated_text": "string",    // Translated text
  "source_language": "string",    // Source language code
  "target_language": "string"     // Target language code
}
```

**Example:**
```bash
curl -X POST http://localhost:8077/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_language": "en",
    "target_language": "es",
    "model_name": "base"
  }'
```

**Response:**
```json
{
  "translated_text": "Hola, ¿cómo estás?",
  "source_language": "en",
  "target_language": "es"
}
```

**Constraints:**
- Maximum input length: 512 tokens
- Text is automatically truncated if too long
- Empty strings return 400 Bad Request

---

### TTS Service

**Base URL:** `http://localhost:8078`

Text-to-Speech service using Coqui XTTS models.

#### Endpoints

##### 1. Health Check
**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "tts",
  "device": "cpu"
}
```

##### 2. List Models
**Endpoint:** `GET /models`

**Response:**
```json
{
  "models": ["base"]
}
```

##### 3. Synthesize Speech
**Endpoint:** `POST /synthesize`

**Request:**
```json
{
  "text": "string",               // Text to synthesize
  "language": "string",           // Target language code
  "model_name": "string"          // Optional: Model name (default: "base")
}
```

**Response:**
```json
{
  "audio": "string"               // Base64 encoded audio (float32 PCM)
}
```

**Example:**
```bash
curl -X POST http://localhost:8078/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hola, ¿cómo estás?",
    "language": "es",
    "model_name": "base"
  }' | jq -r '.audio' | base64 -d > output.raw
```

**Audio Output:**
- Format: Raw PCM audio as numpy float32 array
- Encoding: Base64
- Sample Rate: Model-dependent (typically 22050 Hz or 24000 Hz)

---

## Client Implementation Guide

### Python Client Example

```python
import requests
import base64
import numpy as np
from typing import Literal

class MultimodalTranslationClient:
    def __init__(self, gateway_url: str = "http://localhost:8075"):
        """
        Initialize the translation client.
        
        Args:
            gateway_url: Base URL of the API Gateway
        """
        self.gateway_url = gateway_url
        
    def translate(
        self,
        input_data: str,
        input_type: Literal["text", "audio"],
        source_language: str,
        target_language: str,
        output_type: Literal["text", "audio"],
        asr_model: str = "base",
        nmt_model: str = "base",
        tts_model: str = "base"
    ) -> dict:
        """
        Perform translation through the API Gateway.
        
        Args:
            input_data: Text string or base64 encoded audio
            input_type: Type of input ("text" or "audio")
            source_language: Source language code
            target_language: Target language code
            output_type: Desired output type ("text" or "audio")
            asr_model: ASR model name (default: "base")
            nmt_model: NMT model name (default: "base")
            tts_model: TTS model name (default: "base")
            
        Returns:
            Dictionary with translation response
        """
        payload = {
            "input": input_data,
            "input_type": input_type,
            "source_language": source_language,
            "target_language": target_language,
            "output_type": output_type,
            "asr_model": asr_model,
            "nmt_model": nmt_model,
            "tts_model": tts_model
        }
        
        response = requests.post(
            f"{self.gateway_url}/translate",
            json=payload,
            timeout=120  # Allow time for model loading and processing
        )
        response.raise_for_status()
        return response.json()
    
    def translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str,
        nmt_model: str = "base"
    ) -> str:
        """
        Translate text to text.
        
        Returns:
            Translated text string
        """
        result = self.translate(
            input_data=text,
            input_type="text",
            source_language=source_language,
            target_language=target_language,
            output_type="text",
            nmt_model=nmt_model
        )
        return result["output"]
    
    def translate_audio_to_text(
        self,
        audio_file_path: str,
        source_language: str,
        target_language: str,
        asr_model: str = "base",
        nmt_model: str = "base"
    ) -> str:
        """
        Transcribe and translate audio to text.
        
        Args:
            audio_file_path: Path to audio file (should be float32 PCM at 16kHz)
            
        Returns:
            Translated text string
        """
        # Load and encode audio
        audio_array = np.fromfile(audio_file_path, dtype=np.float32)
        audio_base64 = base64.b64encode(audio_array.tobytes()).decode('utf-8')
        
        result = self.translate(
            input_data=audio_base64,
            input_type="audio",
            source_language=source_language,
            target_language=target_language,
            output_type="text",
            asr_model=asr_model,
            nmt_model=nmt_model
        )
        return result["output"]
    
    def translate_text_to_audio(
        self,
        text: str,
        source_language: str,
        target_language: str,
        output_file_path: str,
        nmt_model: str = "base",
        tts_model: str = "base"
    ):
        """
        Translate text and synthesize speech.
        
        Args:
            text: Input text to translate
            output_file_path: Path where to save the audio output
        """
        result = self.translate(
            input_data=text,
            input_type="text",
            source_language=source_language,
            target_language=target_language,
            output_type="audio",
            nmt_model=nmt_model,
            tts_model=tts_model
        )
        
        # Decode and save audio
        audio_bytes = base64.b64decode(result["output"])
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        audio_array.tofile(output_file_path)
    
    def translate_audio_to_audio(
        self,
        input_audio_path: str,
        output_audio_path: str,
        source_language: str,
        target_language: str,
        asr_model: str = "base",
        nmt_model: str = "base",
        tts_model: str = "base"
    ):
        """
        Full pipeline: transcribe, translate, and synthesize.
        
        Args:
            input_audio_path: Path to input audio file
            output_audio_path: Path where to save output audio
        """
        # Load and encode input audio
        audio_array = np.fromfile(input_audio_path, dtype=np.float32)
        audio_base64 = base64.b64encode(audio_array.tobytes()).decode('utf-8')
        
        result = self.translate(
            input_data=audio_base64,
            input_type="audio",
            source_language=source_language,
            target_language=target_language,
            output_type="audio",
            asr_model=asr_model,
            nmt_model=nmt_model,
            tts_model=tts_model
        )
        
        # Decode and save output audio
        audio_bytes = base64.b64decode(result["output"])
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        audio_array.tofile(output_audio_path)
    
    def health_check(self) -> dict:
        """Check health of all services."""
        response = requests.get(f"{self.gateway_url}/health")
        response.raise_for_status()
        return response.json()


# Usage Examples
if __name__ == "__main__":
    # Initialize client
    client = MultimodalTranslationClient("http://localhost:8075")
    
    # Check health
    health = client.health_check()
    print("Health:", health)
    
    # Example 1: Text to Text
    translated = client.translate_text(
        text="Hello, how are you?",
        source_language="en",
        target_language="es"
    )
    print(f"Translated: {translated}")
    
    # Example 2: Audio to Text (assuming you have an audio file)
    # translated = client.translate_audio_to_text(
    #     audio_file_path="input_audio.raw",
    #     source_language="en",
    #     target_language="es"
    # )
    # print(f"Translated: {translated}")
    
    # Example 3: Text to Audio
    # client.translate_text_to_audio(
    #     text="Hello, how are you?",
    #     source_language="en",
    #     target_language="es",
    #     output_file_path="output_audio.raw"
    # )
    # print("Audio saved to output_audio.raw")
```

---

### JavaScript/TypeScript Client Example

```typescript
import axios, { AxiosInstance } from 'axios';

interface TranslationRequest {
  input: string;
  input_type: 'text' | 'audio';
  source_language: string;
  target_language: string;
  output_type: 'text' | 'audio';
  asr_model?: string;
  nmt_model?: string;
  tts_model?: string;
}

interface TranslationResponse {
  output: string;
  output_type: 'text' | 'audio';
  source_language: string;
  target_language: string;
}

class MultimodalTranslationClient {
  private client: AxiosInstance;
  
  constructor(gatewayUrl: string = 'http://localhost:8075') {
    this.client = axios.create({
      baseURL: gatewayUrl,
      timeout: 120000, // 2 minutes
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }
  
  async translate(request: TranslationRequest): Promise<TranslationResponse> {
    const response = await this.client.post<TranslationResponse>(
      '/translate',
      request
    );
    return response.data;
  }
  
  async translateText(
    text: string,
    sourceLanguage: string,
    targetLanguage: string,
    nmtModel: string = 'base'
  ): Promise<string> {
    const result = await this.translate({
      input: text,
      input_type: 'text',
      source_language: sourceLanguage,
      target_language: targetLanguage,
      output_type: 'text',
      nmt_model: nmtModel
    });
    return result.output;
  }
  
  async healthCheck(): Promise<any> {
    const response = await this.client.get('/health');
    return response.data;
  }
}

// Usage
const client = new MultimodalTranslationClient();

(async () => {
  // Check health
  const health = await client.healthCheck();
  console.log('Health:', health);
  
  // Translate text
  const translated = await client.translateText(
    'Hello, how are you?',
    'en',
    'es'
  );
  console.log('Translated:', translated);
})();
```

---

### cURL Examples

```bash
# Text to Text Translation
curl -X POST http://localhost:8075/translate \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, how are you?",
    "input_type": "text",
    "source_language": "en",
    "target_language": "es",
    "output_type": "text"
  }'

# Check Health
curl http://localhost:8075/health

# List Available Models (individual service)
curl http://localhost:8076/models  # ASR
curl http://localhost:8077/models  # NMT
curl http://localhost:8078/models  # TTS
```

---

## Error Handling

### HTTP Status Codes

| Status Code | Meaning | Common Causes |
|-------------|---------|---------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid input format, empty text, malformed base64 |
| 404 | Not Found | Model not found, invalid endpoint |
| 500 | Internal Server Error | Service error, model loading failure |
| 503 | Service Unavailable | Downstream service not responding |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors and Solutions

#### 1. Model Not Found (404)
```json
{
  "detail": "Model 'invalid_model' not found or failed to load"
}
```
**Solution:** Use `GET /models` endpoint to list available models.

#### 2. Service Unavailable (503)
```json
{
  "detail": "NMT service failed: Connection refused"
}
```
**Solution:** Check that all services are running with `bash check_services.sh`.

#### 3. Invalid Audio Format (400)
```json
{
  "detail": "Invalid audio format: cannot convert to float32 array"
}
```
**Solution:** Ensure audio is properly formatted as float32 PCM at 16kHz and correctly base64 encoded.

#### 4. Empty Text (400)
```json
{
  "detail": "Text cannot be empty"
}
```
**Solution:** Provide non-empty text for translation or synthesis.

### Client Error Handling Example

```python
import requests

def safe_translate(client, text, source_lang, target_lang):
    try:
        result = client.translate_text(text, source_lang, target_lang)
        return result
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print("Model not found. Using default model.")
            # Retry with default model
        elif e.response.status_code == 503:
            print("Service unavailable. Please check services.")
        else:
            print(f"Error: {e.response.json()['detail']}")
    except requests.exceptions.Timeout:
        print("Request timed out. Service may be loading models.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    return None
```

---

## Language Codes

The system uses language codes compatible with the underlying models. Below are common examples:

### Common Language Codes

| Language | Code | NLLB Format |
|----------|------|-------------|
| English | en | eng_Latn |
| Spanish | es | spa_Latn |
| French | fr | fra_Latn |
| German | de | deu_Latn |
| Italian | it | ita_Latn |
| Portuguese | pt | por_Latn |
| Chinese (Simplified) | zh | zho_Hans |
| Japanese | ja | jpn_Jpan |
| Korean | ko | kor_Hang |
| Arabic | ar | arb_Arab |
| Russian | ru | rus_Cyrl |
| Hindi | hi | hin_Deva |

**Note:** The NMT service automatically handles language code conversion. You can use either format:
- Simple codes: `"en"`, `"es"`, `"fr"`
- NLLB codes: `"eng_Latn"`, `"spa_Latn"`, `"fra_Latn"`

For a complete list of supported languages, refer to:
- **Whisper:** Supports 99+ languages for ASR
- **NLLB:** Supports 200+ languages for translation
- **XTTS:** Supports 17+ languages for TTS

---

## Audio Format Specifications

### Input Audio Format (ASR)

The ASR service expects audio in the following format:

- **Format:** Raw PCM audio
- **Data Type:** float32 (32-bit floating point)
- **Sample Rate:** 16,000 Hz (16 kHz)
- **Channels:** Mono (1 channel)
- **Encoding:** Base64
- **Value Range:** -1.0 to 1.0 (normalized)

### Output Audio Format (TTS)

The TTS service produces audio in the following format:

- **Format:** Raw PCM audio
- **Data Type:** float32 (32-bit floating point)
- **Sample Rate:** Model-dependent (typically 22,050 Hz or 24,000 Hz)
- **Channels:** Mono (1 channel)
- **Encoding:** Base64
- **Value Range:** -1.0 to 1.0 (normalized)

### Audio Conversion Examples

#### Convert WAV to Required Format (Python)

```python
import numpy as np
import librosa

def prepare_audio_for_asr(input_file: str) -> str:
    """Convert audio file to format required by ASR service."""
    # Load audio file and resample to 16kHz
    audio, sr = librosa.load(input_file, sr=16000, mono=True)
    
    # Ensure float32 format and normalize
    audio = audio.astype(np.float32)
    
    # Encode to base64
    import base64
    audio_base64 = base64.b64encode(audio.tobytes()).decode('utf-8')
    
    return audio_base64

def save_audio_from_tts(audio_base64: str, output_file: str):
    """Save audio from TTS service to WAV file."""
    import base64
    import soundfile as sf
    
    # Decode base64
    audio_bytes = base64.b64decode(audio_base64)
    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
    
    # Save as WAV (assuming 22050 Hz sample rate)
    sf.write(output_file, audio_array, 22050)
```

#### Convert WAV to Required Format (ffmpeg)

```bash
# Convert any audio file to 16kHz mono float32 PCM
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f f32le -acodec pcm_f32le output.raw

# Encode to base64
base64 -w 0 output.raw > output.b64

# Decode base64 audio from TTS and convert to WAV
echo "<base64_audio>" | base64 -d > output.raw
ffmpeg -f f32le -ar 22050 -ac 1 -i output.raw output.wav
```

---

## Rate Limiting and Performance

### Expected Latency

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| Text-to-Text | 0.5-2s | Depends on text length |
| Audio-to-Text | 2-5s | Depends on audio length |
| Text-to-Audio | 3-8s | Depends on text length |
| Audio-to-Audio | 5-15s | Full pipeline |

**Note:** First request may be slower due to model loading (lazy loading pattern).

### Recommendations

1. **Implement Timeouts:** Set client timeouts to at least 120 seconds for full pipeline requests
2. **Handle Model Loading:** First request to each service loads the model and takes longer
3. **Parallel Requests:** Services are stateless and can handle multiple requests concurrently
4. **Caching:** Consider caching translations for frequently used phrases
5. **Monitoring:** Use health check endpoints to monitor service availability

---

## Troubleshooting

### Service Won't Start

```bash
# Check if ports are in use
lsof -i :8000  # API Gateway
lsof -i :8001  # ASR
lsof -i :8002  # NMT
lsof -i :8003  # TTS

# View service logs
tail -f services/_logs/api_gateway.log
tail -f services/_logs/asr.log
tail -f services/_logs/nmt.log
tail -f services/_logs/tts.log
```

### Service Not Responding

```bash
# Check service status
bash services/check_services.sh

# Restart services
bash services/stop_all_services.sh
bash services/start_all_services.sh
```

### Model Not Found

```bash
# List available models for each service
curl http://localhost:8076/models  # ASR models
curl http://localhost:8077/models  # NMT models
curl http://localhost:8078/models  # TTS models

# Check model directory structure
ls -la services/asr/models/
ls -la services/nmt/models/
ls -la services/tts/models/
```

---

## Advanced Configuration

### Distributed Deployment

Deploy services on different servers:

```bash
# Server 1: ASR Service
export ASR_PORT=8001
cd services/asr
uv run python service.py

# Server 2: NMT Service
export NMT_PORT=8002
cd services/nmt
uv run python service.py

# Server 3: TTS Service
export TTS_PORT=8003
cd services/tts
uv run python service.py

# Server 4: API Gateway
export GATEWAY_PORT=8000
export ASR_SERVICE_URL="http://server1:8001"
export NMT_SERVICE_URL="http://server2:8002"
export TTS_SERVICE_URL="http://server3:8003"
cd services/api_gateway
uv run python api.py
```

### Load Balancing

Use a reverse proxy (nginx, HAProxy) to load balance requests:

```nginx
upstream api_gateway {
    server gateway1:8000;
    server gateway2:8000;
    server gateway3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_gateway;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Support and Resources

### Documentation Files
- `services/README.md` - Service overview and quick start
- `services/API_DOCUMENTATION.md` - This file (comprehensive API reference)

### Service Scripts
- `services/start_all_services.sh` - Start all services
- `services/stop_all_services.sh` - Stop all services
- `services/check_services.sh` - Check service status
- `services/restart_service.sh` - Restart specific service
- `services/view_logs.sh` - View service logs

### Log Files
- `services/_logs/api_gateway.log`
- `services/_logs/asr.log`
- `services/_logs/nmt.log`
- `services/_logs/tts.log`

---

## Version Information

- **API Version:** 1.0.0
- **FastAPI:** 0.119.0
- **Python:** 3.9+
- **Last Updated:** October 2025

---

## License and Credits

This system uses the following open-source technologies:

- **Whisper:** OpenAI (Apache 2.0 License)
- **NLLB:** Meta AI (CC-BY-NC License)
- **Coqui XTTS:** Coqui AI (MPL 2.0 License)
- **FastAPI:** Sebastián Ramírez (MIT License)
- **Transformers:** Hugging Face (Apache 2.0 License)

---

**End of Documentation**
