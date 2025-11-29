"""
API Gateway - Orchestrates multimodal translation pipeline

Required packages:
- fastapi
- uvicorn
- httpx (for async HTTP requests)
"""
import os
os.environ['OPENSSL_CONF'] = ''

# Also try to prevent SSL context creation issues
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

import logging
from typing import Literal, Optional
import base64

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

from data_logger import EvaluationDataLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Multimodal Translation API Gateway", version="1.0.0")

# Initialize evaluation data logger
data_logger = EvaluationDataLogger()

# Add CORS middleware to allow requests from the web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development. In production, specify exact origins.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Service URLs from environment variables with validation
# If individual service URLs are not set, construct them from individual port settings
ASR_PORT = os.getenv("ASR_PORT", "8076")
NMT_PORT = os.getenv("NMT_PORT", "8077")
TTS_PORT = os.getenv("TTS_PORT", "8078")
EVALUATION_API_PORT = os.getenv("EVALUATION_API_PORT", "8079")

ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", f"http://localhost:{ASR_PORT}")
NMT_SERVICE_URL = os.getenv("NMT_SERVICE_URL", f"http://localhost:{NMT_PORT}")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", f"http://localhost:{TTS_PORT}")
EVALUATION_SERVICE_URL = os.getenv("EVALUATION_SERVICE_URL", f"http://localhost:{EVALUATION_API_PORT}")


class TranslationRequest(BaseModel):
    """Request model for translation pipeline"""
    input: str  # text or base64 encoded audio
    input_type: Literal["text", "audio"]
    source_language: str
    target_language: str
    output_type: Literal["text", "audio"]
    asr_model: Optional[str] = "base"  # Model selection for ASR
    nmt_model: Optional[str] = "base"  # Model selection for NMT
    tts_model: Optional[str] = "base"  # Model selection for TTS
    save_for_evaluation: Optional[bool] = False  # Save data for evaluation


class TranslationResponse(BaseModel):
    """Response model for translation pipeline"""
    output: str  # text or base64 encoded audio
    output_type: Literal["text", "audio"]
    source_language: str
    target_language: str
    # Intermediate results from pipeline steps
    transcribed_text: Optional[str] = None  # From ASR (if audio input)
    translated_text: Optional[str] = None  # From NMT (always available)
    output_audio: Optional[str] = None  # From TTS (if audio output)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if all services are reachable
    services_status = {}
    
    async with httpx.AsyncClient(
    timeout=5.0, 
    verify=False, 
    trust_env=False,
    transport=httpx.AsyncHTTPTransport(retries=0)  # Don't create SSL context for HTTP
    ) as client:
        for service_name, service_url in [
            ("asr", ASR_SERVICE_URL),
            ("nmt", NMT_SERVICE_URL),
            ("tts", TTS_SERVICE_URL)
        ]:
            try:
                response = await client.get(f"{service_url}/health")
                services_status[service_name] = response.json()
            except Exception as e:
                services_status[service_name] = {"status": "unhealthy", "error": str(e)}
    
    return {
        "status": "healthy",
        "service": "api_gateway",
        "downstream_services": services_status
    }


@app.get("/models/asr")
async def get_asr_models():
    """Get available ASR models"""
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False, trust_env=False) as client:
            response = await client.get(f"{ASR_SERVICE_URL}/models")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching ASR models: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch ASR models: {str(e)}"
        )


@app.get("/models/nmt")
async def get_nmt_models():
    """Get available NMT models"""
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False, trust_env=False) as client:
            response = await client.get(f"{NMT_SERVICE_URL}/models")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching NMT models: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch NMT models: {str(e)}"
        )


@app.get("/models/tts")
async def get_tts_models():
    """Get available TTS models"""
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False, trust_env=False) as client:
            response = await client.get(f"{TTS_SERVICE_URL}/models")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching TTS models: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch TTS models: {str(e)}"
        )


@app.get("/evaluations")
async def list_evaluations():
    """Proxy: List all evaluation executions from Evaluation Service"""
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False, trust_env=False) as client:
            response = await client.get(f"{EVALUATION_SERVICE_URL}/executions")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching evaluations: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch evaluations: {str(e)}"
        )


@app.get("/evaluations/{execution_id}")
async def get_evaluation(execution_id: str):
    """Proxy: Get detailed evaluation execution from Evaluation Service"""
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False, trust_env=False) as client:
            response = await client.get(f"{EVALUATION_SERVICE_URL}/executions/{execution_id}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching evaluation {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch evaluation: {str(e)}"
        )


@app.get("/evaluations/{execution_id}/languages/{language}")
async def get_evaluation_language(execution_id: str, language: str):
    """Proxy: Get language-specific evaluation results from Evaluation Service"""
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False, trust_env=False) as client:
            response = await client.get(
                f"{EVALUATION_SERVICE_URL}/executions/{execution_id}/languages/{language}"
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching language {language} for {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch language results: {str(e)}"
        )


@app.get("/evaluations/{execution_id}/languages/{language}/files/{filename}")
async def get_evaluation_language_file(execution_id: str, language: str, filename: str):
    """Proxy: Download language-specific files from Evaluation Service"""
    try:
        async with httpx.AsyncClient(timeout=30.0, verify=False, trust_env=False) as client:
            response = await client.get(
                f"{EVALUATION_SERVICE_URL}/executions/{execution_id}/languages/{language}/files/{filename}"
            )
            response.raise_for_status()
            return Response(
                content=response.content,
                media_type=response.headers.get("content-type", "application/octet-stream"),
                headers={"Content-Disposition": response.headers.get("content-disposition", "")}
            )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching file {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch file: {str(e)}"
        )


@app.get("/evaluations/{execution_id}/files/{filename}")
async def get_evaluation_file(execution_id: str, filename: str):
    """Proxy: Download execution-level files from Evaluation Service"""
    try:
        async with httpx.AsyncClient(timeout=30.0, verify=False, trust_env=False) as client:
            response = await client.get(
                f"{EVALUATION_SERVICE_URL}/executions/{execution_id}/files/{filename}"
            )
            response.raise_for_status()
            return Response(
                content=response.content,
                media_type=response.headers.get("content-type", "application/octet-stream"),
                headers={"Content-Disposition": response.headers.get("content-disposition", "")}
            )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching file {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch file: {str(e)}"
        )


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Main translation endpoint that orchestrates the pipeline
    
    Routing logic:
    - If input_type == "audio": call ASR service first
    - Always call NMT service for translation
    - If output_type == "audio": call TTS service last
    
    Args:
        request: TranslationRequest with input data, configuration, and model selections
        
    Returns:
        TranslationResponse with translated output
    """
    try:
        logger.info(
            f"Processing translation: {request.input_type} -> {request.output_type}, "
            f"{request.source_language} -> {request.target_language}, "
            f"models: ASR={request.asr_model}, NMT={request.nmt_model}, TTS={request.tts_model}"
        )
        
        text_to_translate = request.input
        transcribed_text = None
        translated_text = None
        output_audio = None
        
        # Step 1: Speech-to-Text (if input is audio)
        if request.input_type == "audio":
            logger.info(f"Step 1: Transcribing audio with model '{request.asr_model}'...")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    asr_response = await client.post(
                        f"{ASR_SERVICE_URL}/transcribe",
                        json={
                            "audio": request.input,
                            "source_language": request.source_language,
                            "model_name": request.asr_model
                        }
                    )
                    asr_response.raise_for_status()
                    asr_data = asr_response.json()
                    text_to_translate = asr_data["text"]
                    transcribed_text = text_to_translate  # Store for response
                    logger.info(f"Transcription: {text_to_translate[:100]}...")
                    
                except httpx.HTTPError as e:
                    logger.error(f"ASR service error: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"ASR service failed: {str(e)}"
                    )
        
        # Step 2: Translation (always)
        logger.info(f"Step 2: Translating text with model '{request.nmt_model}'...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                nmt_response = await client.post(
                    f"{NMT_SERVICE_URL}/translate",
                    json={
                        "text": text_to_translate,
                        "source_language": request.source_language,
                        "target_language": request.target_language,
                        "model_name": request.nmt_model
                    }
                )
                nmt_response.raise_for_status()
                nmt_data = nmt_response.json()
                translated_text = nmt_data["translated_text"]  # Store for response
                logger.info(f"Translation: {translated_text[:100]}...")
                
            except httpx.HTTPError as e:
                logger.error(f"NMT service error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"NMT service failed: {str(e)}"
                )
        
        # Step 3: Text-to-Speech (if output is audio)
        if request.output_type == "audio":
            logger.info(f"Step 3: Synthesizing speech with model '{request.tts_model}'...")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    tts_response = await client.post(
                        f"{TTS_SERVICE_URL}/synthesize",
                        json={
                            "text": translated_text,
                            "language": request.target_language,
                            "model_name": request.tts_model
                        }
                    )
                    tts_response.raise_for_status()
                    tts_data = tts_response.json()
                    final_output = tts_data["audio"]
                    output_audio = final_output  # Store for response
                    logger.info("Speech synthesis completed")
                    
                except httpx.HTTPError as e:
                    logger.error(f"TTS service error: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"TTS service failed: {str(e)}"
                    )
        else:
            final_output = translated_text
        
        logger.info("Translation pipeline completed successfully")
        
        # Save for evaluation if requested
        if request.save_for_evaluation:
            try:
                # Determine task type
                task_type = f"{request.input_type}_to_{request.output_type}"
                
                # Decode audio if present
                source_audio_bytes = None
                target_audio_bytes = None
                
                if request.input_type == "audio":
                    source_audio_bytes = base64.b64decode(request.input)
                
                if request.output_type == "audio":
                    target_audio_bytes = base64.b64decode(final_output)
                
                # Save the data
                sample_id = data_logger.save_translation(
                    task_type=task_type,
                    source_text=request.input if request.input_type == "text" else None,
                    target_text=translated_text,
                    source_lang=request.source_language,
                    target_lang=request.target_language,
                    source_audio=source_audio_bytes,
                    target_audio=target_audio_bytes,
                    transcribed_text=transcribed_text,
                    models_used={
                        "asr_model": request.asr_model if request.input_type == "audio" else None,
                        "nmt_model": request.nmt_model,
                        "tts_model": request.tts_model if request.output_type == "audio" else None
                    }
                )
                logger.info(f"Saved evaluation data with ID: {sample_id}")
            except Exception as e:
                logger.error(f"Failed to save evaluation data: {e}")
                # Don't fail the request if evaluation saving fails
        
        return TranslationResponse(
            output=final_output,
            output_type=request.output_type,
            source_language=request.source_language,
            target_language=request.target_language,
            transcribed_text=transcribed_text,
            translated_text=translated_text,
            output_audio=output_audio
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation pipeline failed: {str(e)}"
        )


if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("GATEWAY_PORT", 8075))
    
    logger.info(f"Starting API Gateway on port {port}...")
    logger.info("=" * 60)
    logger.info("Service Configuration:")
    logger.info(f"  ASR Service URL: {ASR_SERVICE_URL}")
    logger.info(f"  NMT Service URL: {NMT_SERVICE_URL}")
    logger.info(f"  TTS Service URL: {TTS_SERVICE_URL}")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )