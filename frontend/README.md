# Multimodal Translation Pipeline Dashboard

A modern, dark-themed dashboard for testing and evaluating multilingual translation pipelines. This application combines both frontend and backend in a single package, supporting text-to-text, text-to-speech, speech-to-text, and speech-to-speech translation workflows.

## Features

🌟 **Modern UI**: Dark-themed dashboard built with React, TypeScript, and Tailwind CSS
🔄 **Translation Modes**: Free translation and evaluation mode with metrics
🎯 **Multiple Pipeline Types**: 
   - Text → Text (NMT)
   - Text → Speech (TTS)
   - Speech → Text (ASR)
   - Speech → Speech (Full pipeline)

📊 **Analytics**: Performance metrics, BLEU scores, COMET scores, latency tracking
🔧 **Pipeline Management**: Configure and compare different model architectures
📁 **File Support**: Text files, audio files (WAV, MP3, M4A, FLAC), and JSON batches

## Quick Start

### Development

```bash
# Install dependencies
npm install

# Start both frontend and backend in development mode
npm run dev
```

This will start:
- Frontend (React + Vite) on http://localhost:3000
- Backend API (Express) on http://localhost:3001

### Production Build

```bash
# Build the application
npm run build

# Start production server
npm start
```

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/pipelines` - List available pipeline configurations
- `GET /api/pipelines/:id` - Get specific pipeline details
- `POST /api/translate` - Single translation request
- `POST /api/translate/batch` - Batch translation request

For detailed API documentation, see [API_DOCUMENTATION.md](../API_DOCUMENTATION.md).

For Model Manager integration details, see [MODEL_MANAGER_INTEGRATION.md](../MODEL_MANAGER_INTEGRATION.md).

## Pipeline Configurations

The application comes with three pre-configured pipelines:

### Baseline Pipeline
- ASR: Whisper Base
- NMT: OPUS-MT  
- TTS: eSpeak NG

### Advanced Pipeline  
- ASR: Whisper Large
- NMT: mBART Large
- TTS: Tacotron 2

### Experimental Pipeline
- ASR: Wav2Vec2 XL
- NMT: NLLB-200
- TTS: Bark

## Translation Modes

### Free Translation
Basic translation without reference comparison. Returns translated output with processing time.

### Evaluation Mode
Compares translation against provided reference text. Returns:
- BLEU scores
- COMET scores  
- Word Error Rate (WER) for ASR
- Mean Opinion Score (MOS) for TTS
- Latency and throughput metrics

## File Upload Support

- **Text Files**: .txt, .json (for batch processing)
- **Audio Files**: .wav, .mp3, .m4a, .flac (max 100MB)
- **Batch JSON**: Array of translation requests

## Architecture

```
src/
├── client/          # React frontend
│   ├── components/  # UI components
│   ├── App.tsx      # Main app component
│   └── main.tsx     # React entry point
├── server/          # Express backend
│   └── index.ts     # API server
└── shared/          # Shared types and utilities
    └── types.ts     # TypeScript interfaces
```

## Mock Data

The current implementation uses mock responses for demonstration. In production, you would replace the mock translation functions with actual calls to your model manager/inference service.

## Development Notes

- The frontend uses Vite's proxy to route `/api` requests to the backend
- Both frontend and backend are written in TypeScript
- Tailwind CSS provides the dark theme styling
- Lucide React icons for the UI
- Express.js with multer for file uploads

## Future Enhancements

- Real model integration with Python backend
- User authentication and session management
- Model fine-tuning interface
- Advanced analytics and reporting
- Real-time translation streaming
- Custom pipeline builder UI