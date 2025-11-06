"""
TTS Service - Text-to-Speech using Coqui XTTS

Required packages:
- fastapi
- uvicorn
- TTS (Coqui TTS)
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
from TTS.api import TTS as CoquiTTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="TTS Service", version="1.0.0")

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

# Default reference audio for XTTS voice cloning
DEFAULT_REFERENCE_AUDIO = None

def get_default_reference_audio() -> Optional[Path]:
    """Get path to default reference audio file for voice cloning"""
    global DEFAULT_REFERENCE_AUDIO
    
    if DEFAULT_REFERENCE_AUDIO is not None:
        return DEFAULT_REFERENCE_AUDIO
    
    # Look for reference audio in order of preference
    ref_dir = Path(__file__).parent / "reference_audio"
    possible_refs = [
        ref_dir / "female_en.wav",
        ref_dir / "male_en.wav",
        ref_dir / "default.wav",
    ]
    
    for ref_path in possible_refs:
        if ref_path.exists():
            logger.info(f"Found default reference audio: {ref_path}")
            DEFAULT_REFERENCE_AUDIO = ref_path
            return DEFAULT_REFERENCE_AUDIO
    
    # No reference audio found
    logger.warning("No default reference audio found. Voice cloning will not work.")
    DEFAULT_REFERENCE_AUDIO = False  # Mark as checked but not found
    return None


class SynthesizeRequest(BaseModel):
    """Request model for synthesis"""
    text: str
    language: str
    model_name: Optional[str] = "base"  # Model selection


class SynthesizeResponse(BaseModel):
    """Response model for synthesis"""
    audio: str  # base64 encoded audio


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


def get_xtts_language_code(lang_code: str) -> str:
    """
    Convert language code to XTTS-compatible format.
    XTTS expects simple 2-letter codes like 'en', 'es', 'fr', etc.
    
    Args:
        lang_code: Input language code (can be 'en', 'eng_Latn', etc.)
        
    Returns:
        XTTS-compatible language code
    """
    # Mapping of common language codes to XTTS format
    language_map = {
        # Simple codes (pass through)
        'en': 'en',
        'es': 'es',
        'efi': 'efi',
        'ibo': 'ibo',
        'xho': 'xho',
        'swa': 'swa',
        # NLLB format to XTTS
        'eng_Latn': 'en',
        'spa_Latn': 'es',
        'efi_Latn': 'efi',
        'ibo_Latn': 'ibo',
        'xho_Latn': 'xho',
        'swh_Latn': 'swa',
    }
    
    # Try exact match first
    if lang_code in language_map:
        return language_map[lang_code]
    
    # Try extracting first 2 letters
    simple_code = lang_code[:2].lower()
    if simple_code in language_map:
        return language_map[simple_code]
    
    # Default to English if unknown
    logger.warning(f"Unknown language code: {lang_code}, defaulting to 'en'")
    return 'en'


def load_xtts_model(model_name: str):
    """
    Load XTTS model (lazy loading with caching)
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        TTS model instance
    """
    if model_name in loaded_models:
        logger.info(f"Using cached model: {model_name}")
        return loaded_models[model_name]
    
    try:
        logger.info(f"Loading XTTS model: {model_name} on {DEVICE}...")
        
        # Load model from models directory
        model_dir = get_models_directory() / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Check for required files
        config_path = model_dir / "config.json"
        model_path = model_dir / "model.pth"
        
        if not config_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Required model files not found in {model_dir}. "
                f"Expected config.json and model.pth"
            )
        
        # Load XTTS model directly (not through TTS API wrapper)
        # This is necessary for custom trained models
        config = XttsConfig()
        config.load_json(str(config_path))
        
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True, use_deepspeed=False)
        model.to(DEVICE)
        
        # Cache both config and model
        loaded_models[model_name] = (config, model)
        
        logger.info(f"XTTS model '{model_name}' loaded successfully on {DEVICE}")
        
        return config, model
        
    except Exception as e:
        logger.error(f"Error loading model '{model_name}': {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tts", "device": DEVICE}


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


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text using XTTS
    
    Args:
        request: SynthesizeRequest containing text, language, and model selection
        
    Returns:
        SynthesizeResponse with base64 encoded audio
    """
    try:
        logger.info(
            f"Synthesizing text for language {request.language}: {request.text[:50]}..., "
            f"model: {request.model_name}"
        )
        
        # Load model (lazy loading with caching)
        try:
            config, model = load_xtts_model(request.model_name)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{request.model_name}' not found or failed to load: {str(e)}"
            )
        
        # Validate input
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )
        
        # Generate speech using XTTS inference
        # Convert language code to XTTS-compatible format
        xtts_language = get_xtts_language_code(request.language)
        logger.info(f"Using XTTS language code: {xtts_language} (from {request.language})")
        
        # Get reference audio for voice cloning
        ref_audio = get_default_reference_audio()
        if not ref_audio or not ref_audio.exists():
            logger.error("XTTS model requires voice cloning but no reference audio available")
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=(
                    "This XTTS model requires a reference speaker audio file for voice cloning. "
                    "No default reference audio found. Please run: "
                    "'python create_reference_audio.py' in the tts directory to set up a default voice."
                )
            )
        
        logger.info(f"Using reference audio for voice cloning: {ref_audio}")
        
        try:
            # Compute speaker latents from reference audio
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=str(ref_audio)
            )
            
            # Generate speech
            out = model.inference(
                text=request.text,
                language=xtts_language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.7,
                length_penalty=1.0,
                repetition_penalty=5.0,
                top_k=50,
                top_p=0.85,
            )
            
            # Check if inference returned valid output
            if out is None:
                logger.error(f"Model inference returned None for text: {request.text[:100]}")
                raise ValueError("Model inference returned None. This usually means the model is not properly loaded or initialized.")
            
            # Extract waveform
            if isinstance(out, dict):
                wav = out.get("wav")
            else:
                # Sometimes inference returns the wav directly
                wav = out
            
            # Validate wav output
            if wav is None:
                logger.error(f"No 'wav' key in inference output. Output keys: {out.keys() if isinstance(out, dict) else type(out)}")
                raise ValueError("Inference did not produce audio output")
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"TTS synthesis failed: {str(e)}"
            )
        
        # Convert to numpy array if not already
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)
        
        # Convert to float32 format
        wav = wav.astype(np.float32)
        
        # Encode to base64
        audio_bytes = wav.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        logger.info(f"Synthesis completed, audio size: {len(audio_bytes)} bytes")
        
        return SynthesizeResponse(audio=audio_base64)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {str(e)}"
        )


if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("TTS_PORT", 8078))
    
    logger.info(f"Starting TTS service on port {port}...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )