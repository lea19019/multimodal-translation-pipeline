from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
import time
import uuid
from datetime import datetime

from models import (
    HealthResponse, LoadModelsRequest, LoadModelsResponse, 
    TranslationRequest, TranslationResponse,
    TranscriptionResponse, SynthesisResponse,
    BatchRequest, BatchResponse, ErrorResponse,
    DummyDataGenerator
)
from config import settings, PIPELINE_CONFIGS

# Initialize FastAPI app
app = FastAPI(
    title="Model Manager API",
    description="Multimodal Translation Pipeline Model Manager",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3003"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to track loaded models
loaded_models = {
    "asr": ["whisper-base"],
    "nmt": ["opus-mt"], 
    "tts": ["espeak-ng"]
}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the Model Manager is healthy and models are loaded."""
    return DummyDataGenerator.generate_health_response()

@app.get("/models/status")
async def get_model_status():
    """Get the status of all loaded models."""
    return {
        "loadedModels": loaded_models,
        "totalModels": sum(len(models) for models in loaded_models.values()),
        "memoryUsage": "4.2GB",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/models/load", response_model=LoadModelsResponse)
async def load_models(request: LoadModelsRequest):
    """Load specific models into memory."""
    # Simulate model loading
    await asyncio.sleep(0.5)  # Simulate loading time
    
    loaded_model_info = []
    for model_config in request.models:
        model_type = model_config.get("type")
        model_name = model_config.get("name")
        
        if model_type not in loaded_models:
            loaded_models[model_type] = []
        
        if model_name not in loaded_models[model_type]:
            loaded_models[model_type].append(model_name)
        
        loaded_model_info.append({
            "type": model_type,
            "name": model_name,
            "status": "loaded",
            "memoryUsage": f"{hash(model_name) % 4 + 1}.{hash(model_name) % 9}GB",
            "loadTime": 10.0 + (hash(model_name) % 20)
        })
    
    return LoadModelsResponse(
        success=True,
        loadedModels=loaded_model_info
    )

@app.post("/translate/text", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Perform neural machine translation."""
    try:
        # Validate model is loaded
        if request.model not in loaded_models.get("nmt", []):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not loaded. Available models: {loaded_models.get('nmt', [])}"
            )
        
        # Simulate processing time
        await asyncio.sleep(0.1 + len(request.text) * 0.01)
        
        return DummyDataGenerator.generate_translation_response(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    model: str = Form(...),
    language: str = Form(default="auto"),
    task: str = Form(default="transcribe"),
    audio: UploadFile = File(...),
    options: Optional[str] = Form(default="{}")
):
    """Convert speech to text using ASR models."""
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
        
        # Simulate processing time based on file size
        processing_time = 0.5 + (len(audio_content) / 1000000)  # 1 second per MB
        await asyncio.sleep(min(processing_time, 3.0))  # Cap at 3 seconds for demo
        
        return DummyDataGenerator.generate_transcription_response(audio.filename or "audio.wav")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_speech(request: dict):
    """Generate speech from text using TTS models."""
    try:
        model = request.get("model")
        text = request.get("text")
        language = request.get("language", "en")
        
        if not model or not text:
            raise HTTPException(status_code=400, detail="Model and text are required")
        
        # Validate model is loaded
        if model not in loaded_models.get("tts", []):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' is not loaded. Available models: {loaded_models.get('tts', [])}"
            )
        
        # Simulate processing time
        await asyncio.sleep(0.8 + len(text) * 0.02)
        
        return DummyDataGenerator.generate_synthesis_response(text)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch/process", response_model=BatchResponse)
async def process_batch(request: BatchRequest):
    """Process multiple requests in a single batch."""
    try:
        results = []
        start_time = time.time()
        
        for req in request.requests:
            req_start = time.time()
            
            try:
                # Process based on request type
                req_type = req.get("type")
                
                if req_type == "translate":
                    # Create translation request
                    trans_req = TranslationRequest(
                        model=req.get("model"),
                        sourceLang=req.get("sourceLang"),
                        targetLang=req.get("targetLang"),
                        text=req.get("text")
                    )
                    response = DummyDataGenerator.generate_translation_response(trans_req)
                    result_data = response.dict()["translation"]
                    
                elif req_type == "transcribe":
                    # Simulate transcription
                    response = DummyDataGenerator.generate_transcription_response("batch_audio.wav")
                    result_data = response.dict()["transcription"]
                    
                elif req_type == "synthesize":
                    # Simulate synthesis
                    text = req.get("text", "Default text")
                    response = DummyDataGenerator.generate_synthesis_response(text)
                    result_data = response.dict()["synthesis"]
                    
                else:
                    raise ValueError(f"Unknown request type: {req_type}")
                
                results.append({
                    "id": req.get("id"),
                    "success": True,
                    "result": result_data,
                    "processingTime": time.time() - req_start
                })
                
            except Exception as e:
                results.append({
                    "id": req.get("id"),
                    "success": False,
                    "error": str(e),
                    "processingTime": time.time() - req_start
                })
        
        total_time = time.time() - start_time
        successful_requests = sum(1 for r in results if r["success"])
        
        return BatchResponse(
            success=True,
            batchId=request.batchId,
            results=results,
            summary={
                "totalRequests": len(request.requests),
                "successfulRequests": successful_requests,
                "failedRequests": len(request.requests) - successful_requests,
                "totalProcessingTime": total_time,
                "parallelWorkers": min(len(request.requests), 4)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipelines")
async def get_pipelines():
    """Get available pipeline configurations."""
    return list(PIPELINE_CONFIGS.values())

@app.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get specific pipeline configuration."""
    if pipeline_id not in PIPELINE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    
    config = PIPELINE_CONFIGS[pipeline_id].copy()
    config["status"] = "active"
    config["lastUpdated"] = datetime.now().isoformat()
    
    return config

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "type": type(exc).__name__,
                "details": str(exc)
            },
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    print(f"🚀 Model Manager starting on {settings.host}:{settings.port}")
    print(f"📋 Loaded models: {loaded_models}")
    print(f"🏭 Available pipelines: {list(PIPELINE_CONFIGS.keys())}")

# Shutdown event  
@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Model Manager shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )