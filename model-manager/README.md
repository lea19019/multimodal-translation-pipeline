# Model Manager

Python-based Model Manager for the Multimodal Translation Pipeline.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or using conda
conda create -n model-manager python=3.9
conda activate model-manager
pip install -r requirements.txt
```

## Running the Server

```bash
# Development mode (with auto-reload)
python run.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /models/status` - Model loading status  
- `POST /models/load` - Load specific models

### Translation Services
- `POST /translate/text` - Text translation
- `POST /transcribe` - Speech-to-text
- `POST /synthesize` - Text-to-speech
- `POST /batch/process` - Batch processing

### Pipeline Management
- `GET /pipelines` - List pipeline configurations
- `GET /pipelines/{id}` - Get specific pipeline

## Configuration

Create a `.env` file to override default settings:

```env
HOST=0.0.0.0
PORT=8000
DEBUG=true
MODEL_CACHE_DIR=./models
MAX_BATCH_SIZE=32
DEFAULT_ASR_MODEL=whisper-base
DEFAULT_NMT_MODEL=opus-mt
DEFAULT_TTS_MODEL=espeak-ng
```

## Docker Deployment

```bash
# Build image
docker build -t model-manager .

# Run container
docker run -p 8000:8000 -v ./models:/models model-manager
```

## API Documentation

When running, visit:
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Current Status

This is a **placeholder implementation** that returns dummy data for all translation modalities. It provides:

- ✅ Full API compatibility with the documented interface
- ✅ Realistic response formats and timing
- ✅ Proper error handling and validation
- ✅ Support for all modalities (text↔text, text↔speech, speech↔text, speech↔speech)
- ❌ Actual ML model integration (placeholder only)

## Next Steps

To integrate real models:

1. Replace dummy data generators with actual model calls
2. Add proper model loading and caching
3. Implement GPU memory management
4. Add model-specific preprocessing/postprocessing
5. Optimize for production performance