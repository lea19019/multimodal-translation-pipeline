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
```

## Services

- **API Gateway** (Port 8075): Orchestrates the translation pipeline
- **ASR Service** (Port 8076): Automatic Speech Recognition using Whisper
- **NMT Service** (Port 8077): Neural Machine Translation using NLLB
- **TTS Service** (Port 8078): Text-to-Speech using XTTS

## Quick Start

### Start All Services

```bash
cd /home/vacl2/multimodal_translation/services
bash start_all_services.sh
```

This will:
- Start all four services in the background
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
```

## Environment Configuration

Services can be configured using environment variables:

### Port Configuration

```bash
export ASR_PORT=8076
export NMT_PORT=8077
export TTS_PORT=8078
export GATEWAY_PORT=8075
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

# View all logs in real-time
tail -f _logs/*.log
```

## API Endpoints

### API Gateway (http://localhost:8075)

- `GET /health` - Health check for all services
- `POST /translate` - Main translation endpoint

Example request:
```json
{
  "input": "Hello, how are you?",
  "input_type": "text",
  "source_language": "en",
  "target_language": "es",
  "output_type": "text",
  "asr_model": "base",
  "nmt_model": "base",
  "tts_model": "base"
}
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
