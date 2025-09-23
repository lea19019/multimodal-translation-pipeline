import os
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Model Configuration
    model_cache_dir: str = "./models"
    max_model_memory: str = "8GB"
    gpu_memory_fraction: float = 0.8
    
    # Processing Configuration
    max_batch_size: int = 32
    parallel_workers: int = 4
    request_timeout: int = 30
    
    # Default Models
    default_asr_model: str = "whisper-base"
    default_nmt_model: str = "opus-mt"
    default_tts_model: str = "espeak-ng"
    
    # Audio Configuration
    default_sample_rate: int = 22050
    max_audio_duration: int = 600  # 10 minutes
    supported_audio_formats: List[str] = ["wav", "mp3", "m4a", "flac"]
    
    # Text Configuration
    max_text_length: int = 512
    supported_languages: List[str] = ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "ar"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Pipeline configurations
PIPELINE_CONFIGS = {
    "baseline": {
        "id": "baseline",
        "name": "Baseline Pipeline",
        "description": "Basic translation pipeline for general use",
        "models": {
            "asr": {"name": "whisper-base", "version": "v1", "device": "cpu"},
            "nmt": {"name": "opus-mt", "version": "v1", "device": "cpu"},
            "tts": {"name": "espeak-ng", "version": "v1", "device": "cpu"}
        },
        "performance": {
            "latency": "low",
            "accuracy": "medium",
            "resource_usage": "low"
        }
    },
    "advanced": {
        "id": "advanced", 
        "name": "Advanced Pipeline",
        "description": "High-accuracy pipeline with latest models",
        "models": {
            "asr": {"name": "whisper-large", "version": "v3", "device": "cuda:0"},
            "nmt": {"name": "mbart-large", "version": "50", "device": "cuda:0"},
            "tts": {"name": "tacotron2", "version": "v1", "device": "cuda:1"}
        },
        "performance": {
            "latency": "medium",
            "accuracy": "high", 
            "resource_usage": "high"
        }
    },
    "experimental": {
        "id": "experimental",
        "name": "Experimental Pipeline", 
        "description": "Cutting-edge models for research",
        "models": {
            "asr": {"name": "wav2vec2-xl", "version": "v1", "device": "cuda:0"},
            "nmt": {"name": "nllb-200", "version": "v1", "device": "cuda:1"},
            "tts": {"name": "bark", "version": "v1", "device": "cuda:2"}
        },
        "performance": {
            "latency": "high",
            "accuracy": "very_high",
            "resource_usage": "very_high"
        }
    }
}