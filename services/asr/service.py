"""
ASR Service - Automatic Speech Recognition using OpenAI Whisper

Required packages:
- fastapi
- uvicorn
- transformers
- torch
- numpy
"""

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ASR Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for loaded models - lazy loading pattern
loaded_models = {}

# CPU-only execution
DEVICE = "cpu"


class TranscribeRequest(BaseModel):
    """Request model for transcription"""
    audio: str  # base64 encoded audio
    source_language: Optional[str] = "en"
    model_name: Optional[str] = "base"  # Model selection


class TranscribeResponse(BaseModel):
    """Response model for transcription"""
    text: str
    language: str


class ModelsResponse(BaseModel):
    """Response model for available models"""
    models: List[str]


def get_models_directory() -> Path:
    """Get the models directory path"""
    return Path(__file__).parent / "checkpoints"


def list_available_models() -> List[str]:
    """List all available models in the models directory"""
    models_dir = get_models_directory()
    
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return []
    
    # List subdirectories in models folder
    models = [d.name for d in models_dir.iterdir() if d.is_dir()]
    return sorted(models)


def load_whisper_model(model_name: str):
    """
    Load Whisper model and processor (lazy loading with caching)
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        tuple: (processor, model)
    """
    if model_name in loaded_models:
        logger.info(f"Using cached model: {model_name}")
        return loaded_models[model_name]
    
    try:
        logger.info(f"Loading Whisper model: {model_name} on {DEVICE}...")
        
        # Load model from models directory
        model_dir = get_models_directory() / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load processor and model
        processor = WhisperProcessor.from_pretrained(str(model_dir))
        model = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
        
        # Force CPU execution
        model = model.to(DEVICE)
        model.eval()
        
        # Cache the loaded model
        loaded_models[model_name] = (processor, model)
        
        logger.info(f"Whisper model '{model_name}' loaded successfully on {DEVICE}")
        
        return processor, model
        
    except Exception as e:
        logger.error(f"Error loading model '{model_name}': {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "asr", "device": DEVICE}


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """
    Get list of available models
    
    Returns:
        ModelsResponse with list of available model names
    """
    try:
        models = list_available_models()
        logger.info(f"Available models: {models}")
        return ModelsResponse(models=models)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """
    Transcribe audio to text using Whisper
    
    Args:
        request: TranscribeRequest containing base64 encoded audio and model selection
        
    Returns:
        TranscribeResponse with transcribed text and language
    """
    try:
        logger.info(
            f"Transcribing audio for language: {request.source_language}, "
            f"model: {request.model_name}"
        )
        
        # Load model (lazy loading with caching)
        try:
            processor, model = load_whisper_model(request.model_name)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{request.model_name}' not found or failed to load: {str(e)}"
            )
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(request.audio)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 audio: {str(e)}"
            )
        
        # Load audio (assuming it's a numpy array or raw audio)
        # Note: In production, you'd want to handle different audio formats
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid audio format: {str(e)}"
            )
        
        # Process audio
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(inputs)
        
        # Decode transcription
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        logger.info(f"Transcription completed: {transcription[:50]}...")
        
        return TranscribeResponse(
            text=transcription,
            language=request.source_language
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("ASR_PORT", 8076))
    
    logger.info(f"Starting ASR service on port {port}...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )