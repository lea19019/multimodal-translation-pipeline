# Multimodal Translation Services

This directory contains microservices for a multimodal translation pipeline that handles audio-to-audio, text-to-text, and mixed-mode translation.

## Architecture

```
┌─────────────────┐
│  API Gateway    │  Port 8075
│   (api.py)      │
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
┌───▼───┐ ┌──▼───┐ ┌──▼───┐    │
│  ASR  │ │ NMT  │ │ TTS  │    │
│ 8076  │ │ 8077 │ │ 8078 │    │
└───────┘ └──────┘ └──────┘    │
                                │
                         (Direct text path)

┌─────────────────┐
│ Evaluation API  │  Port 8079
│  (service.py)   │
└─────────────────┘
```

## Services

- **API Gateway** (Port 8075): Orchestrates the translation pipeline
- **ASR Service** (Port 8076): Automatic Speech Recognition using Whisper
- **NMT Service** (Port 8077): Neural Machine Translation using NLLB
- **TTS Service** (Port 8078): Text-to-Speech using XTTS
- **Evaluation API** (Port 8079): Serves evaluation results and metrics

## Quick Start

### Start All Services

```bash
cd /home/vacl2/multimodal_translation/services
bash start_all_services.sh
```

This will:
- Start all five services in the background
- Create log files in `services/_logs/` directory
- Save process IDs for management

### Check Service Status

```bash
bash check_services.sh
```

This will show:
- Which services are running
- Process IDs and ports
- Health check status for each service

### Stop All Services

```bash
bash stop_all_services.sh
```

This will gracefully stop all running services.

## Individual Service Management

You can also run services individually:

```bash
# ASR Service
cd asr
uv run python service.py

# NMT Service
cd nmt
uv run python service.py

# TTS Service
cd tts
uv run python service.py

# API Gateway
cd api_gateway
uv run python api.py

# Evaluation API
cd evaluation
uv run python service.py
```

## Environment Configuration

Services can be configured using environment variables:

### Port Configuration

```bash
export ASR_PORT=8076
export NMT_PORT=8077
export TTS_PORT=8078
export GATEWAY_PORT=8075
export EVALUATION_API_PORT=8079
```

### Service URL Override

For distributed deployments, you can override the service URLs:

```bash
export ASR_SERVICE_URL="http://asr-server:8076"
export NMT_SERVICE_URL="http://nmt-server:8077"
export TTS_SERVICE_URL="http://tts-server:8078"
```

## Logs

All service logs are stored in `services/_logs/`:

```bash
# View all logs
ls -la _logs/

# Follow a specific service log
tail -f _logs/api_gateway.log
tail -f _logs/asr.log
tail -f _logs/nmt.log
tail -f _logs/tts.log
tail -f _logs/evaluation_api.log

# View all logs in real-time
tail -f _logs/*.log
```

## API Endpoints

### API Gateway (http://localhost:8075)

**Translation Pipeline:**
- `GET /health` - Health check for all services
- `POST /translate` - Main translation endpoint
- `GET /models/asr` - List available ASR models
- `GET /models/nmt` - List available NMT models
- `GET /models/tts` - List available TTS models

**Evaluation Results (proxied to Evaluation Service):**
- `GET /evaluations` - List all evaluation executions
- `GET /evaluations/{execution_id}` - Get detailed execution results
- `GET /evaluations/{execution_id}/languages/{language}` - Get language-specific results
- `GET /evaluations/{execution_id}/languages/{language}/files/{filename}` - Download files
- `GET /evaluations/{execution_id}/files/{filename}` - Download execution-level files

Example translation request:
```json
{
  "input": "Hello, how are you?",
  "input_type": "text",
  "source_language": "en",
  "target_language": "es",
  "output_type": "text",
  "asr_model": "base",
  "nmt_model": "base",
  "tts_model": "base",
  "save_for_evaluation": false
}
```

**Note:** Frontend should only call the API Gateway (port 8075). Evaluation endpoints are automatically proxied to the Evaluation Service.

### Evaluation Service (http://localhost:8079)

Backend service (not called directly by frontend) that serves evaluation results:

- `GET /health` - Health check
- `GET /executions` - List all evaluation executions
- `GET /executions/{execution_id}` - Get detailed execution results
- `GET /executions/{execution_id}/languages/{language}` - Get language-specific results
- `GET /executions/{execution_id}/languages/{language}/files/{filename}` - Download visualization/result files
- `GET /executions/{execution_id}/files/{filename}` - Download execution-level files (manifest.json, overall_summary.json)

Example response from `/executions`:
```json
[
  {
    "execution_id": "eval_20251119_105437",
    "timestamp": "2025-11-19T10:54:37",
    "nmt_model": "nllb-200-distilled-600M",
    "tts_model": "xtts_v2",
    "metrics": ["bleu", "comet", "wer"],
    "languages": {
      "swahili": {"total_samples": 50, "valid_samples": 48},
      "igbo": {"total_samples": 50, "valid_samples": 47}
    },
    "total_samples": 100,
    "total_valid_samples": 95,
    "overall_summary": {...}
  }
]
```

### Individual Services

Each service has:
- `GET /health` - Service health check
- `GET /models` - List available models
- `POST /<service-specific>` - Main service endpoint
  - ASR: `/transcribe`
  - NMT: `/translate`
  - TTS: `/synthesize`

## Troubleshooting

### Services won't start

1. Check if ports are already in use:
   ```bash
   lsof -i :8075
   lsof -i :8076
   lsof -i :8077
   lsof -i :8078
   lsof -i :8079
   ```

2. Check service logs:
   ```bash
   cat _logs/<service_name>.log
   ```

3. Verify dependencies are installed in each service directory

### Services not responding

1. Check service status:
   ```bash
   bash check_services.sh
   ```

2. Check if models are loaded properly (may take time on first run)

3. Restart services:
   ```bash
   bash stop_all_services.sh
   bash start_all_services.sh
   ```

## Development

### Adding a New Service

1. Create a new directory under `services/`
2. Implement the service with FastAPI
3. Add health check endpoint at `GET /health`
4. Update `start_all_services.sh` to include the new service
5. Update this README

### Requirements

Each service should have:
- `service.py` (or equivalent main file)
- `requirements.txt` or `pyproject.toml`
- Model files in `models/` directory
- Configuration in `.env` file (optional)

## Notes

- All services run on CPU by default
- Models are lazy-loaded on first request
- Services cache loaded models for efficiency
- Use `uv` for Python environment management
