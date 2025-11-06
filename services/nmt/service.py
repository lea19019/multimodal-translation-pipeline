"""
NMT Service - Neural Machine Translation using Meta NLLB

Required packages:
- fastapi
- uvicorn
- transformers
- torch
"""

import logging
import os
from pathlib import Path
from typing import Optional, List

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="NMT Service", version="1.0.0")

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


class TranslateRequest(BaseModel):
    """Request model for translation"""
    text: str
    source_language: str
    target_language: str
    model_name: Optional[str] = "base"  # Model selection


class TranslateResponse(BaseModel):
    """Response model for translation"""
    translated_text: str
    source_language: str
    target_language: str


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


def load_nllb_model(model_name: str):
    """
    Load NLLB model and tokenizer (lazy loading with caching)
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        tuple: (tokenizer, model)
    """
    if model_name in loaded_models:
        logger.info(f"Using cached model: {model_name}")
        return loaded_models[model_name]
    
    try:
        logger.info(f"Loading NLLB model: {model_name} on {DEVICE}...")
        
        # Load model from models directory
        model_dir = get_models_directory() / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
        
        # Force CPU execution
        model = model.to(DEVICE)
        model.eval()
        
        # Cache the loaded model
        loaded_models[model_name] = (tokenizer, model)
        
        logger.info(f"NLLB model '{model_name}' loaded successfully on {DEVICE}")
        
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Error loading model '{model_name}': {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "nmt", "device": DEVICE}


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


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """
    Translate text from source to target language using NLLB
    
    Args:
        request: TranslateRequest containing text, language codes, and model selection
        
    Returns:
        TranslateResponse with translated text
    """
    try:
        logger.info(
            f"Translating from {request.source_language} to {request.target_language}: "
            f"{request.text[:50]}..., model: {request.model_name}"
        )
        
        # Load model (lazy loading with caching)
        try:
            tokenizer, model = load_nllb_model(request.model_name)
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
        
        # Convert language codes to NLLB format if needed
        def get_nllb_lang_code(lang_code: str) -> str:
            """Convert simple language code to NLLB format"""
            # Map of common codes to NLLB format
            lang_map = {
                'en': 'eng_Latn',
                'es': 'spa_Latn',
                'efi': 'efi_Latn',
                'ibo': 'ibo_Latn',
                'xho': 'xho_Latn',
                'swa': 'swh_Latn',
            }
            
            # If already in NLLB format, return as is
            if '_' in lang_code:
                return lang_code
            
            # Otherwise, look up in map or try adding _Latn
            return lang_map.get(lang_code, f"{lang_code}_Latn")
        
        # Set source language for tokenizer
        src_lang = get_nllb_lang_code(request.source_language)
        tgt_lang = get_nllb_lang_code(request.target_language)
        
        # Set the source language
        tokenizer.src_lang = src_lang
        
        # Tokenize input text
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)
        
        # Get target language token ID
        try:
            # Try to get the language token ID
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        except:
            # Fallback: let the model decide
            forced_bos_token_id = None
            logger.warning(f"Could not find token ID for {tgt_lang}, using default")
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512
            )
        
        # Decode translation
        translated_text = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        logger.info(f"Translation completed: {translated_text[:50]}...")
        
        return TranslateResponse(
            translated_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("NMT_PORT", 8077))
    
    logger.info(f"Starting NMT service on port {port}...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )