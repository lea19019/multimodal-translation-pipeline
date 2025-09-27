# Code Comments and Inline Documentation

This document explains the key code sections with detailed inline comments for better understanding.

## Frontend Code Analysis

### TranslationInterface.tsx - Main UI Component

```typescript
/**
 * Main translation interface component
 * Handles all user interactions and manages translation state
 */
const TranslationInterface: React.FC = () => {
  // State management for translation configuration
  const [type, setType] = useState<TranslationType>('text-to-text');
  const [sourceLang, setSourceLang] = useState('en');
  const [targetLang, setTargetLang] = useState('es');
  
  // Input state - handles text, file, and audio inputs
  const [input, setInput] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  
  // Translation result and loading state
  const [result, setResult] = useState<TranslationResponse | null>(null);
  const [loading, setLoading] = useState(false);

  /**
   * Determines if the selected translation type requires audio input
   * Used to show/hide recording controls and file upload options
   */
  const requiresAudioInput = type === 'speech-to-text' || type === 'speech-to-speech';

  /**
   * Main translation submission handler
   * Validates input, formats request, and calls API
   */
  const handleSubmit = async () => {
    // Input validation
    if (!selectedPipeline) {
      alert('Please select a pipeline first');
      return;
    }

    // Determine input source and validate
    let inputData: string | FormData;
    if (file) {
      // File upload mode
      const formData = new FormData();
      formData.append('file', file);
      formData.append('request', JSON.stringify({
        type, sourceLang, targetLang, 
        pipelineId: selectedPipeline.id,
        mode, reference
      }));
      inputData = formData;
    } else if (recordedBlob) {
      // Audio recording mode
      const formData = new FormData();
      formData.append('file', recordedBlob, 'recording.webm');
      formData.append('request', JSON.stringify({
        type, sourceLang, targetLang,
        pipelineId: selectedPipeline.id, 
        mode, reference
      }));
      inputData = formData;
    } else {
      // Text input mode
      inputData = JSON.stringify({
        type, sourceLang, targetLang, input,
        pipelineId: selectedPipeline.id,
        mode, reference
      });
    }

    // API call with error handling
    setLoading(true);
    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        ...(typeof inputData === 'string' ? {
          headers: { 'Content-Type': 'application/json' },
          body: inputData
        } : {
          body: inputData // FormData sets its own Content-Type
        })
      });

      const data = await response.json();
      if (data.data?.success) {
        setResult(data.data);
      } else {
        throw new Error(data.data?.error || 'Translation failed');
      }
    } catch (error) {
      console.error('Translation error:', error);
      alert('Translation failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Audio recording management
   * Handles MediaRecorder API for live audio capture
   */
  const startRecording = async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });
      
      // Find supported audio format
      const mimeTypes = ['audio/webm', 'audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg'];
      const supportedMimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type));
      
      // Initialize recorder
      const recorder = new MediaRecorder(stream, 
        supportedMimeType ? { mimeType: supportedMimeType } : {}
      );
      
      const chunks: BlobPart[] = [];

      // Handle recorded data
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      // Handle recording completion
      recorder.onstop = () => {
        const mimeType = recorder.mimeType || 'audio/webm';
        const blob = new Blob(chunks, { type: mimeType });
        setRecordedBlob(blob);
        stream.getTracks().forEach(track => track.stop());
        setIsRecording(false);
      };

      // Start recording
      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
      
    } catch (error) {
      console.error('Recording failed:', error);
      setRecordingError('Microphone access denied or not available');
    }
  };

  /**
   * Result display logic
   * Handles different output types (text, audio, error)
   */
  const renderOutput = () => {
    if (!result) return null;

    // Audio output types (TTS, speech-to-speech)
    if (type === 'text-to-speech' || type === 'speech-to-speech') {
      return (
        <div className="bg-dark-900 border border-dark-600 rounded-md p-4">
          <div className="flex items-center space-x-4">
            <Play className="h-6 w-6 text-blue-400" />
            <div>
              <div className="text-sm font-medium">Audio Output</div>
              <div className="text-xs text-slate-400">Generated speech file</div>
            </div>
            <button className="btn-secondary ml-auto">
              <Download className="h-4 w-4" />
            </button>
          </div>
        </div>
      );
    }

    // Text output types (text-to-text, speech-to-text)
    return (
      <div className="bg-dark-900 border border-dark-600 rounded-md p-4">
        <p className="text-slate-100">
          {result.result?.translatedText || result.output || 'No translation result'}
        </p>
      </div>
    );
  };
};
```

## API Gateway Code Analysis

### index.ts - Express Server

```typescript
/**
 * Express server setup with CORS and middleware configuration
 */
const app = express();
const port = 3003; // API Gateway port

// Initialize translation service (handles Model Manager communication)
const translationService = new TranslationService();

// Middleware setup
app.use(cors()); // Enable cross-origin requests from frontend
app.use(express.json({ limit: '50mb' })); // Parse JSON bodies up to 50MB

// Configure multer for file uploads
const upload = multer({ 
  storage: multer.memoryStorage(), // Store files in memory
  limits: { fileSize: 100 * 1024 * 1024 } // 100MB file size limit
});

/**
 * Main translation endpoint
 * Handles both JSON and multipart/form-data requests
 */
app.post('/api/translate', upload.single('file'), async (req, res) => {
  try {
    console.log('📥 Received translation request');
    console.log('Body keys:', Object.keys(req.body));
    console.log('File:', req.file ? 'Yes' : 'None');

    // Parse request data (handles both JSON and form-data)
    let requestData;
    if (req.body.request) {
      // Form-data request (with file upload)
      requestData = JSON.parse(req.body.request);
      if (req.file) {
        requestData.input = req.file.buffer; // Attach file buffer
      }
    } else {
      // JSON request (text input)
      requestData = req.body;
    }

    // Ensure pipeline ID is set (defaults to 'baseline')
    requestData.pipelineId = requestData.pipelineId || 'baseline';
    
    console.log(`🔄 Processing ${requestData.type} translation: ${requestData.sourceLang} → ${requestData.targetLang}`);

    // Process translation through service layer
    const result = await translationService.processTranslation(requestData);
    
    // Format successful response
    const response: ApiResponse<any> = {
      data: {
        success: true,
        result,
        metadata: {
          processingTime: result.processingTime || 0,
          pipelineId: requestData.pipelineId,
          type: requestData.type,
          ...result.metadata
        },
        timestamp: new Date().toISOString()
      },
      message: 'Translation completed successfully'
    };
    
    res.json(response);
    
  } catch (error) {
    console.error('Translation processing failed:', error);
    
    // Format error response
    const errorResponse: ApiResponse<any> = {
      data: {
        success: false,
        error: {
          code: 'PROCESSING_ERROR',
          message: error.message,
          type: error.constructor.name,
          details: error.details || {}
        }
      },
      message: 'Translation failed'
    };
    
    res.status(500).json(errorResponse);
  }
});

/**
 * Health check endpoint
 * Verifies API Gateway and Model Manager connectivity
 */
app.get('/api/health', async (_req, res) => {
  try {
    const health = await translationService.healthCheck();
    res.json(health);
  } catch (error) {
    res.status(500).json({ 
      status: 'error', 
      message: 'Health check failed',
      error: error.message 
    });
  }
});

/**
 * Pipeline configuration endpoint
 * Returns available translation pipelines from Model Manager
 */
app.get('/api/pipelines', async (_req, res) => {
  try {
    const pipelines = await translationService.getPipelines();
    res.json({ data: pipelines });
  } catch (error) {
    res.status(500).json({ 
      error: 'Failed to fetch pipelines',
      message: error.message 
    });
  }
});
```

### translation-service.ts - Service Layer

```typescript
/**
 * Translation service layer
 * Handles request transformation and Model Manager communication
 */
class TranslationService {
  private modelManagerClient: ModelManagerClient;

  constructor() {
    // Initialize Model Manager client with retry configuration
    this.modelManagerClient = new ModelManagerClient('http://localhost:8000');
  }

  /**
   * Main translation processing method
   * Routes requests to appropriate handlers based on type
   */
  async processTranslation(request: TranslationRequest) {
    console.log(`Processing ${request.type} translation`);

    // Get pipeline configuration from Model Manager
    const pipeline = await this.getPipeline(request.pipelineId || 'baseline');
    
    if (!pipeline) {
      throw new Error(`Pipeline '${request.pipelineId}' not found`);
    }

    // Route to type-specific handler
    switch (request.type) {
      case 'text-to-text':
        return await this.handleTextToText(request, pipeline);
      case 'text-to-speech': 
        return await this.handleTextToSpeech(request, pipeline);
      case 'speech-to-text':
        return await this.handleSpeechToText(request, pipeline);
      case 'speech-to-speech':
        return await this.handleSpeechToSpeech(request, pipeline);
      default:
        throw new Error(`Unsupported translation type: ${request.type}`);
    }
  }

  /**
   * Text-to-text translation handler
   * Simple pass-through to Model Manager NMT endpoint
   */
  private async handleTextToText(request: TranslationRequest, pipeline: PipelineConfig) {
    const translationResponse = await this.modelManagerClient.translateText({
      model: pipeline.models.nmt?.id || 'opus-mt',
      sourceLang: request.sourceLang,
      targetLang: request.targetLang,
      text: request.input as string,
      options: request.options
    });

    return {
      translatedText: translationResponse.translation.text,
      confidence: translationResponse.translation.confidence,
      alternatives: translationResponse.translation.alternatives,
      metadata: translationResponse.metadata
    };
  }

  /**
   * Text-to-speech translation handler
   * 1. Translate text (NMT)
   * 2. Synthesize speech (TTS)
   */
  private async handleTextToSpeech(request: TranslationRequest, pipeline: PipelineConfig) {
    // Step 1: Translate text
    const translationResponse = await this.modelManagerClient.translateText({
      model: pipeline.models.nmt?.id || 'opus-mt',
      sourceLang: request.sourceLang,
      targetLang: request.targetLang,
      text: request.input as string,
      options: request.options
    });

    // Step 2: Synthesize translated text to speech
    const synthResponse = await this.modelManagerClient.synthesizeSpeech({
      model: pipeline.models.tts?.id || 'espeak-ng',
      text: translationResponse.translation.text,
      language: request.targetLang,
      options: request.options
    });

    return {
      translatedText: translationResponse.translation.text,
      audioData: synthResponse.synthesis.audioBase64,
      audioFormat: synthResponse.synthesis.format,
      duration: synthResponse.synthesis.duration,
      metadata: synthResponse.metadata
    };
  }

  /**
   * Speech-to-text translation handler
   * 1. Transcribe audio (ASR)  
   * 2. Translate text (NMT)
   */
  private async handleSpeechToText(request: TranslationRequest, pipeline: PipelineConfig) {
    // Step 1: Transcribe audio to text
    const transcribeResponse = await this.modelManagerClient.transcribeAudio(
      request.input as Buffer,
      {
        model: pipeline.models.asr?.id || 'whisper-base',
        language: request.sourceLang === 'auto' ? 'auto' : request.sourceLang,
        task: 'transcribe',
        options: request.options
      }
    );

    // Step 2: Translate transcribed text
    const translationResponse = await this.modelManagerClient.translateText({
      model: pipeline.models.nmt?.id || 'opus-mt',
      sourceLang: request.sourceLang,
      targetLang: request.targetLang,
      text: transcribeResponse.transcription.text,
      options: request.options
    });

    return {
      transcribedText: transcribeResponse.transcription.text,
      translatedText: translationResponse.translation.text,
      confidence: transcribeResponse.transcription.confidence,
      metadata: {
        asr: transcribeResponse.metadata,
        nmt: translationResponse.metadata
      }
    };
  }

  /**
   * Speech-to-speech translation handler  
   * 1. Transcribe audio (ASR)
   * 2. Translate text (NMT)
   * 3. Synthesize speech (TTS)
   */
  private async handleSpeechToSpeech(request: TranslationRequest, pipeline: PipelineConfig) {
    // Step 1: Transcribe audio
    const transcribeResponse = await this.modelManagerClient.transcribeAudio(
      request.input as Buffer,
      {
        model: pipeline.models.asr?.id || 'whisper-base',
        language: request.sourceLang === 'auto' ? 'auto' : request.sourceLang,
        task: 'transcribe',
        options: request.options
      }
    );

    // Step 2: Translate transcribed text
    const translationResponse = await this.modelManagerClient.translateText({
      model: pipeline.models.nmt?.id || 'opus-mt',
      sourceLang: request.sourceLang,
      targetLang: request.targetLang,
      text: transcribeResponse.transcription.text,
      options: request.options
    });

    // Step 3: Synthesize translated text
    const synthResponse = await this.modelManagerClient.synthesizeSpeech({
      model: pipeline.models.tts?.id || 'espeak-ng',
      text: translationResponse.translation.text,
      language: request.targetLang,
      options: request.options
    });

    return {
      transcribedText: transcribeResponse.transcription.text,
      translatedText: translationResponse.translation.text, 
      audioData: synthResponse.synthesis.audioBase64,
      audioFormat: synthResponse.synthesis.format,
      duration: synthResponse.synthesis.duration,
      confidence: transcribeResponse.transcription.confidence,
      metadata: {
        asr: transcribeResponse.metadata,
        nmt: translationResponse.metadata,
        tts: synthResponse.metadata
      }
    };
  }
}
```

### model-manager-client.ts - HTTP Client

```typescript
/**
 * HTTP client for communicating with Python Model Manager
 * Handles retries, error handling, and request formatting
 */
class ModelManagerClient {
  private baseUrl: string;
  private timeout: number;
  private maxRetries: number;

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.timeout = 30000; // 30 second timeout
    this.maxRetries = 3;   // Retry failed requests 3 times
  }

  /**
   * Generic HTTP request method with retry logic
   */
  private async makeRequest(method: string, endpoint: string, data?: any): Promise<any> {
    const url = `${this.baseUrl}${endpoint}`;
    
    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        console.log(`Making request to Model Manager: ${method} ${url}`);
        
        const response = await fetch(url, {
          method,
          headers: {
            'Content-Type': 'application/json',
          },
          body: data ? JSON.stringify(data) : undefined,
          // Note: Using node-fetch timeout would go here in real implementation
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log(`✅ Model Manager request successful`);
        return result;

      } catch (error) {
        console.error(`Model Manager request attempt ${attempt} failed:`, error.message);
        
        if (attempt === this.maxRetries) {
          throw new ModelManagerError(`Failed after ${this.maxRetries} attempts: ${error.message}`);
        }
        
        // Exponential backoff: wait 1s, 2s, 4s between retries
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt - 1) * 1000));
      }
    }
  }

  /**
   * Text translation endpoint
   */
  async translateText(request: {
    model: string;
    sourceLang: string;
    targetLang: string;
    text: string;
    options?: any;
  }): Promise<any> {
    return this.makeRequest('POST', '/translate/text', request);
  }

  /**
   * Audio transcription endpoint with file upload
   */
  async transcribeAudio(audioBuffer: Buffer, options: {
    model: string;
    language?: string;
    task?: string;
    options?: any;
  }): Promise<any> {
    // Use FormData for file upload
    const FormData = require('form-data');
    const form = new FormData();
    
    form.append('model', options.model);
    form.append('language', options.language || 'auto');
    form.append('task', options.task || 'transcribe');
    form.append('audio', audioBuffer, { 
      filename: 'audio.wav', 
      contentType: 'audio/wav' 
    });
    
    if (options.options) {
      form.append('options', JSON.stringify(options.options));
    }

    return this.makeRequestWithForm('POST', '/transcribe', form);
  }

  /**
   * FormData request handler (for file uploads)
   */
  private async makeRequestWithForm(method: string, endpoint: string, form: any): Promise<any> {
    const url = `${this.baseUrl}${endpoint}`;
    
    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        const response = await fetch(url, {
          method,
          body: form,
          headers: form.getHeaders(), // Let FormData set Content-Type
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();

      } catch (error) {
        if (attempt === this.maxRetries) {
          throw new ModelManagerError(`Failed after ${this.maxRetries} attempts: ${error.message}`);
        }
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt - 1) * 1000));
      }
    }
  }
}

/**
 * Custom error class for Model Manager communication issues
 */
class ModelManagerError extends Error {
  public statusCode: number;
  public details: any;

  constructor(message: string, statusCode: number = 0, details: any = {}) {
    super(message);
    this.name = 'ModelManagerError';
    this.statusCode = statusCode;
    this.details = details;
  }
}
```

## Model Manager Code Analysis

### main.py - FastAPI Application

```python
"""
FastAPI Model Manager
Provides dummy ML model responses for translation pipeline
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
from typing import Optional, Dict, List

# Import custom models and configuration
from models import *
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Model Manager API",
    description="Dummy ML model service for translation pipeline",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI available at /docs
    redoc_url="/redoc"  # ReDoc available at /redoc
)

# CORS middleware - allows frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock loaded models - simulates real model loading
loaded_models: Dict[str, List[str]] = {
    "asr": ["whisper-base", "whisper-large"],    # Speech recognition
    "nmt": ["opus-mt", "mbart-large"],           # Neural machine translation  
    "tts": ["espeak-ng", "tacotron2"]            # Text-to-speech synthesis
}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns system status and loaded models
    """
    return HealthResponse(
        status="healthy",
        loadedModels=loaded_models,
        gpuMemory={
            "total": "24GB",
            "used": "8GB", 
            "available": "16GB"
        },
        timestamp=datetime.now().isoformat()
    )

@app.post("/translate/text", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Neural machine translation endpoint
    Accepts text and returns translated text with metadata
    """
    try:
        # Validate model availability
        if request.model not in loaded_models.get("nmt", []):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not loaded. Available models: {loaded_models.get('nmt', [])}"
            )
        
        logger.info(f"Translating '{request.text}' from {request.sourceLang} to {request.targetLang}")
        
        # Generate dummy response using helper class
        response = DummyDataGenerator.generate_translation_response(
            text=request.text,
            source_lang=request.sourceLang,
            target_lang=request.targetLang,
            model=request.model
        )
        
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.post("/transcribe", response_model=TranscriptionResponse)  
async def transcribe_audio(
    model: str = Form(...),
    language: str = Form(default="auto"),
    task: str = Form(default="transcribe"),
    audio: UploadFile = File(...),
    options: Optional[str] = Form(default="{}")
):
    """
    Speech-to-text transcription endpoint
    Accepts audio files and returns transcribed text
    """
    try:
        # Validate model is loaded
        if model not in loaded_models.get("asr", []):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not loaded. Available models: {loaded_models.get('asr', [])}"
            )
        
        # Validate audio file
        if not audio.content_type or not audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Read audio file (for validation)
        audio_content = await audio.read()
        if len(audio_content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Transcribing audio file: {audio.filename}, size: {len(audio_content)} bytes")
        
        # Generate dummy transcription response
        response = DummyDataGenerator.generate_transcription_response(
            model=model,
            language=language,
            audio_duration=len(audio_content) / 16000  # Estimate duration
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_speech(request: SynthesisRequest):
    """
    Text-to-speech synthesis endpoint
    Accepts text and returns audio data (base64 encoded)
    """
    try:
        # Validate model availability
        if request.model not in loaded_models.get("tts", []):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not loaded. Available models: {loaded_models.get('tts', [])}"
            )
        
        logger.info(f"Synthesizing speech for text: '{request.text}' in {request.language}")
        
        # Generate dummy synthesis response
        response = DummyDataGenerator.generate_synthesis_response(
            text=request.text,
            language=request.language,
            model=request.model
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech synthesis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Synthesis error: {str(e)}")

@app.get("/pipelines")
async def get_pipelines():
    """
    Get available pipeline configurations
    Returns list of translation pipelines with model specifications
    """
    pipelines = [
        {
            "id": "baseline",
            "name": "Baseline Pipeline",
            "description": "Basic translation pipeline for general use",
            "models": {
                "asr": {"id": "whisper-base", "name": "whisper-base", "version": "v1", "type": "asr"},
                "nmt": {"id": "opus-mt", "name": "opus-mt", "version": "v1", "type": "nmt"},
                "tts": {"id": "espeak-ng", "name": "espeak-ng", "version": "v1", "type": "tts"}
            }
        },
        {
            "id": "advanced", 
            "name": "Advanced Pipeline",
            "description": "High-accuracy pipeline with latest models",
            "models": {
                "asr": {"id": "whisper-large", "name": "whisper-large", "version": "v3", "type": "asr"},
                "nmt": {"id": "mbart-large", "name": "mbart-large", "version": "50", "type": "nmt"},
                "tts": {"id": "tacotron2", "name": "tacotron2", "version": "v1", "type": "tts"}
            }
        },
        {
            "id": "experimental",
            "name": "Experimental Pipeline", 
            "description": "Cutting-edge models for research",
            "models": {
                "asr": {"id": "wav2vec2-xl", "name": "wav2vec2-xl", "version": "v1", "type": "asr"},
                "nmt": {"id": "nllb-200", "name": "nllb-200", "version": "v1", "type": "nmt"},
                "tts": {"id": "bark", "name": "bark", "version": "v1", "type": "tts"}
            }
        }
    ]
    
    return pipelines

@app.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """
    Get specific pipeline configuration by ID
    """
    pipelines = await get_pipelines()
    
    for pipeline in pipelines:
        if pipeline["id"] == pipeline_id:
            return pipeline
    
    raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors
    Logs errors and returns standardized error response
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(
        status_code=500,
        detail={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Application startup tasks
    Initialize models, logging, and system resources
    """
    logger.info("🚀 Model Manager starting on %s:%d", settings.host, settings.port)
    logger.info("📋 Loaded models: %s", loaded_models)
    
    # Simulate model loading time
    import asyncio
    await asyncio.sleep(1)
    
    # Log available pipelines
    pipelines = await get_pipelines()
    pipeline_names = [p["id"] for p in pipelines]
    logger.info("🏭 Available pipelines: %s", pipeline_names)

# Shutdown event  
@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown tasks
    Clean up resources and save state
    """
    logger.info("🛑 Model Manager shutting down")
    # In real implementation: unload models, close database connections, etc.

# Development server entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,  # Auto-reload on code changes in debug mode
        log_level="info"
    )
```

### models.py - Data Models and Dummy Responses

```python
"""
Pydantic models for request/response validation
and dummy data generation for testing
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import random
import time
import base64

class TranslationRequest(BaseModel):
    """Request model for text translation"""
    model: str = Field(..., description="Translation model to use")
    sourceLang: str = Field(..., description="Source language code")
    targetLang: str = Field(..., description="Target language code") 
    text: str = Field(..., description="Text to translate")
    options: Optional[Dict[str, Any]] = Field(default={}, description="Additional options")

class TranslationResult(BaseModel):
    """Translation result with confidence and alternatives"""
    text: str = Field(..., description="Translated text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Translation confidence")
    alternatives: List[Dict[str, Any]] = Field(default=[], description="Alternative translations")

class TranslationResponse(BaseModel):
    """Complete translation response with metadata"""
    success: bool = Field(default=True)
    translation: TranslationResult
    metadata: Dict[str, Any] = Field(default={}, description="Processing metadata")

class DummyDataGenerator:
    """
    Helper class for generating realistic dummy responses
    Simulates actual model behavior for testing
    """
    
    @staticmethod
    def generate_translation_response(
        text: str, 
        source_lang: str, 
        target_lang: str, 
        model: str = "opus-mt"
    ) -> TranslationResponse:
        """
        Generate dummy translation response with realistic metadata
        """
        
        # Simulate processing time based on text length
        processing_time = len(text) * 0.02 + random.uniform(0.1, 0.5)
        time.sleep(processing_time)  # Simulate actual processing
        
        # Generate dummy translation
        if model == "mbart-large":
            translation_text = f"[HIGH-QUALITY {source_lang}→{target_lang}]: {text}"
            confidence = random.uniform(0.85, 0.95)  # Higher confidence for better model
        else:
            translation_text = f"[DUMMY TRANSLATION {source_lang}→{target_lang}]: {text}"
            confidence = random.uniform(0.70, 0.90)  # Standard confidence
        
        # Generate alternative translations
        alternatives = [
            {
                "text": f"Alt: {translation_text}",
                "confidence": confidence - 0.1,
                "logProb": -2.1
            }
        ]
        
        # Calculate realistic metadata
        word_count = len(text.split())
        tokens_generated = word_count + random.randint(1, 3)
        
        return TranslationResponse(
            success=True,
            translation=TranslationResult(
                text=translation_text,
                confidence=round(confidence, 2),
                alternatives=alternatives
            ),
            metadata={
                "processingTime": round(processing_time, 2),
                "tokensGenerated": tokens_generated,
                "device": "cpu",  # Could be "gpu" in real implementation
                "modelUsed": model,
                "inputLength": len(text),
                "outputLength": len(translation_text)
            }
        )
    
    @staticmethod
    def generate_transcription_response(
        model: str, 
        language: str, 
        audio_duration: float
    ) -> TranscriptionResponse:
        """
        Generate dummy transcription response
        """
        
        # Simulate processing time (faster than real-time)
        processing_time = audio_duration * 0.3 + random.uniform(0.2, 0.8)
        time.sleep(min(processing_time, 2.0))  # Cap simulation time
        
        # Generate dummy transcription based on model
        if model == "whisper-large":
            transcribed_text = "The weather is beautiful today, and I'm feeling great."
            confidence = random.uniform(0.90, 0.98)
        else:
            transcribed_text = "The weather is beautiful today."
            confidence = random.uniform(0.80, 0.95)
        
        # Detect language (dummy logic)
        detected_language = language if language != "auto" else "en"
        
        return TranscriptionResponse(
            success=True,
            transcription=TranscriptionResult(
                text=transcribed_text,
                confidence=round(confidence, 2),
                language=detected_language,
                segments=[
                    {
                        "start": 0.0,
                        "end": audio_duration,
                        "text": transcribed_text,
                        "confidence": confidence
                    }
                ]
            ),
            metadata={
                "processingTime": round(processing_time, 2),
                "audioDuration": round(audio_duration, 2),
                "modelUsed": model,
                "samplingRate": 16000,
                "device": "cpu"
            }
        )
    
    @staticmethod
    def generate_synthesis_response(
        text: str, 
        language: str, 
        model: str = "espeak-ng"
    ) -> SynthesisResponse:
        """
        Generate dummy speech synthesis response with base64 audio
        """
        
        # Simulate processing time
        processing_time = len(text) * 0.1 + random.uniform(0.5, 2.0)
        time.sleep(min(processing_time, 3.0))  # Cap simulation time
        
        # Generate dummy audio data (minimal WAV file)
        wav_header = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        audio_samples = b'\x00\x01' * (len(text) * 10)  # Scale with text length
        wav_file = wav_header + audio_samples
        
        # Estimate audio duration (roughly 150 words per minute)
        word_count = len(text.split())
        duration = max(1.0, word_count / 2.5)  # ~2.5 words per second
        
        return SynthesisResponse(
            success=True,
            synthesis=SynthesisResult(
                audioBase64=base64.b64encode(wav_file).decode(),
                format="wav",
                samplingRate=22050,
                duration=round(duration, 1),
                byteSize=len(wav_file)
            ),
            metadata={
                "processingTime": round(processing_time, 2),
                "phonemes": f"Phonemes for: {text[:50]}...",
                "device": "cpu",
                "modelVersion": f"{model}-v1.0",
                "textLength": len(text),
                "estimatedDuration": round(duration, 1)
            }
        )
```

This comprehensive documentation should give you a deep understanding of how every part of the system works, making debugging and future development much easier!