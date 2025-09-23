from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import time

class HealthResponse(BaseModel):
    status: str
    loadedModels: Dict[str, List[str]]
    gpuMemory: Dict[str, str]
    timestamp: str

class ModelInfo(BaseModel):
    type: str
    name: str
    status: str
    memoryUsage: str
    loadTime: float

class LoadModelsRequest(BaseModel):
    models: List[Dict[str, Any]]

class LoadModelsResponse(BaseModel):
    success: bool
    loadedModels: List[ModelInfo]

class TranslationRequest(BaseModel):
    model: str
    sourceLang: str
    targetLang: str
    text: str
    options: Optional[Dict[str, Any]] = {}

class TranslationResult(BaseModel):
    text: str
    confidence: float
    logProb: float
    alternatives: Optional[List[Dict[str, Any]]] = []

class TranslationResponse(BaseModel):
    success: bool
    translation: TranslationResult
    metadata: Dict[str, Any]

class TranscriptionSegment(BaseModel):
    text: str
    start: float
    end: float
    confidence: float
    words: Optional[List[Dict[str, Any]]] = []

class TranscriptionResult(BaseModel):
    text: str
    language: str
    confidence: float
    segments: List[TranscriptionSegment]

class TranscriptionResponse(BaseModel):
    success: bool
    transcription: TranscriptionResult
    metadata: Dict[str, Any]

class SynthesisResult(BaseModel):
    audioBase64: str
    format: str
    samplingRate: int
    duration: float
    channels: int

class SynthesisResponse(BaseModel):
    success: bool
    synthesis: SynthesisResult
    metadata: Dict[str, Any]

class BatchRequest(BaseModel):
    batchId: str
    requests: List[Dict[str, Any]]
    options: Optional[Dict[str, Any]] = {}

class BatchResult(BaseModel):
    id: str
    success: bool
    result: Dict[str, Any]
    processingTime: float

class BatchResponse(BaseModel):
    success: bool
    batchId: str
    results: List[BatchResult]
    summary: Dict[str, Any]

class ErrorResponse(BaseModel):
    success: bool = False
    error: Dict[str, Any]
    timestamp: str

# Dummy data generators
class DummyDataGenerator:
    @staticmethod
    def generate_health_response() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            loadedModels={
                "asr": ["whisper-base", "whisper-large"],
                "nmt": ["opus-mt", "mbart-large"],
                "tts": ["espeak-ng", "tacotron2"]
            },
            gpuMemory={
                "total": "24GB",
                "used": "8GB", 
                "available": "16GB"
            },
            timestamp=datetime.now().isoformat()
        )
    
    @staticmethod
    def generate_translation_response(request: TranslationRequest) -> TranslationResponse:
        # Simple dummy translations
        translations = {
            ("en", "es"): {
                "Hello": "Hola",
                "Hello world": "Hola mundo", 
                "Hello, how are you today?": "Hola, ¿cómo estás hoy?",
                "Good morning": "Buenos días",
                "Thank you": "Gracias",
                "How are you?": "¿Cómo estás?"
            },
            ("es", "en"): {
                "Hola": "Hello",
                "Hola mundo": "Hello world",
                "¿Cómo estás?": "How are you?",
                "Buenos días": "Good morning",
                "Gracias": "Thank you"
            },
            ("en", "fr"): {
                "Hello": "Bonjour",
                "Hello world": "Bonjour le monde",
                "Good morning": "Bonjour",
                "Thank you": "Merci"
            }
        }
        
        key = (request.sourceLang, request.targetLang)
        translated_text = translations.get(key, {}).get(
            request.text, 
            f"[DUMMY TRANSLATION {request.sourceLang}→{request.targetLang}]: {request.text}"
        )
        
        processing_time = 0.1 + (len(request.text) * 0.01)  # Simulate processing time
        
        return TranslationResponse(
            success=True,
            translation=TranslationResult(
                text=translated_text,
                confidence=0.85 + (hash(request.text) % 15) / 100,  # Random but consistent confidence
                logProb=-1.5 - (len(request.text) * 0.05),
                alternatives=[
                    {"text": f"Alt: {translated_text}", "confidence": 0.75, "logProb": -2.1}
                ]
            ),
            metadata={
                "processingTime": processing_time,
                "tokensGenerated": len(translated_text.split()),
                "device": "cpu",
                "modelUsed": request.model
            }
        )
    
    @staticmethod
    def generate_transcription_response(filename: str) -> TranscriptionResponse:
        # Dummy transcriptions based on filename patterns or random
        dummy_transcriptions = [
            "Hello, how are you today?",
            "Good morning, this is a test recording.",
            "The weather is beautiful today.",
            "I would like to order some coffee please.",
            "Thank you for using our translation service."
        ]
        
        text = dummy_transcriptions[hash(filename) % len(dummy_transcriptions)]
        words = text.replace(",", "").replace(".", "").split()
        
        # Generate word-level timestamps
        word_segments = []
        current_time = 0.0
        for word in words:
            duration = 0.3 + (len(word) * 0.05)  # Word duration based on length
            word_segments.append({
                "word": word,
                "start": current_time,
                "end": current_time + duration,
                "confidence": 0.9 + (hash(word) % 10) / 100
            })
            current_time += duration + 0.1  # Small pause between words
        
        # Generate segments
        segments = [
            TranscriptionSegment(
                text=text,
                start=0.0,
                end=current_time,
                confidence=0.92,
                words=word_segments
            )
        ]
        
        return TranscriptionResponse(
            success=True,
            transcription=TranscriptionResult(
                text=text,
                language="en",
                confidence=0.92,
                segments=segments
            ),
            metadata={
                "processingTime": 0.5 + (current_time * 0.1),
                "audioLength": current_time,
                "device": "cpu",
                "modelVersion": "whisper-base"
            }
        )
    
    @staticmethod
    def generate_synthesis_response(text: str) -> SynthesisResponse:
        # Generate dummy audio base64 (small WAV header + some data)
        import base64
        
        # Minimal WAV file header + some dummy audio data
        wav_header = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        audio_data = b'\x00\x01' * 100  # Dummy audio samples
        wav_file = wav_header + audio_data
        
        duration = max(1.0, len(text) * 0.1)  # Approximate duration based on text length
        
        return SynthesisResponse(
            success=True,
            synthesis=SynthesisResult(
                audioBase64=base64.b64encode(wav_file).decode(),
                format="wav",
                samplingRate=22050,
                duration=duration,
                channels=1
            ),
            metadata={
                "processingTime": 0.8 + (len(text) * 0.02),
                "phonemes": f"Phonemes for: {text[:20]}...",
                "device": "cpu",
                "modelVersion": "espeak-v1.0"
            }
        )