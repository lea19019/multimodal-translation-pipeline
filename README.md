# Multimodal Translation System

A complete speech-to-speech translation system with microservices architecture and a modern web interface.

## 🌟 Features

- **Text-to-Text Translation**: Translate written text between 200+ languages
- **Audio-to-Text Translation**: Transcribe and translate speech
- **Text-to-Audio Translation**: Generate translated speech from text
- **Audio-to-Audio Translation**: Complete speech-to-speech translation pipeline
- **Web Client**: Modern TypeScript/Next.js interface with audio recording
- **Microservices**: Scalable architecture with independent services
- **REST API**: Well-documented RESTful API

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Client (Next.js)                     │
│              http://localhost:3000                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 API Gateway (Port 8075)                      │
│           Orchestrates Translation Pipeline                  │
└───────────────┬──────────────┬──────────────┬───────────────┘
                │              │              │
        ┌───────▼─────┐  ┌────▼─────┐  ┌────▼──────┐
        │ ASR Service │  │   NMT    │  │    TTS    │
        │  (Whisper)  │  │ (NLLB)   │  │  (XTTS)   │
        │   Port 8076 │  │Port 8077 │  │ Port 8078 │
        └─────────────┘  └──────────┘  └───────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Backend**: Python 3.9+, uv package manager
- **Client**: Node.js 18+, npm

### 1. Start Backend Services

```bash
cd services
bash start_all_services.sh
```

This will start:
- API Gateway on port 8075
- ASR Service on port 8076
- NMT Service on port 8077
- TTS Service on port 8078

### 2. Start Web Client

```bash
cd client
bash start.sh
```

Or manually:
```bash
npm install
npm run dev
```

### 3. Access the Application

Open your browser to **http://localhost:3000**

## 📁 Project Structure

```
multimodal_translation/
├── client/                      # Web client application
│   ├── src/
│   │   ├── app/                # Next.js pages
│   │   ├── components/         # React components
│   │   ├── hooks/              # Custom React hooks
│   │   └── lib/                # API client library
│   ├── package.json
│   ├── README.md               # Client documentation
│   └── start.sh                # Quick start script
│
├── services/                    # Backend microservices
│   ├── api_gateway/            # Main API gateway
│   │   └── api.py
│   ├── asr/                    # Automatic Speech Recognition
│   │   ├── service.py
│   │   ├── whisper_asr.py
│   │   └── models/
│   ├── nmt/                    # Neural Machine Translation
│   │   ├── service.py
│   │   ├── nllb_nmt.py
│   │   └── models/
│   ├── tts/                    # Text-to-Speech
│   │   ├── service.py
│   │   └── models/
│   ├── logs/                   # Service logs
│   ├── API_DOCUMENTATION.md    # Complete API reference
│   ├── README.md               # Services documentation
│   ├── start_all_services.sh
│   ├── stop_all_services.sh
│   └── check_services.sh
│
└── evaluation/                  # Evaluation tools
    └── blaser.py
```

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI
- **ASR**: OpenAI Whisper (via transformers)
- **NMT**: Meta NLLB (No Language Left Behind)
- **TTS**: Coqui XTTS
- **Package Manager**: uv

### Frontend
- **Framework**: Next.js 14
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **HTTP Client**: Axios
- **Audio**: Web Audio API

## 📚 Documentation

- **[API Documentation](services/API_DOCUMENTATION.md)** - Complete REST API reference
- **[Services README](services/README.md)** - Backend services guide
- **[Client README](client/README.md)** - Web client documentation
- **[Client Architecture](client/ARCHITECTURE.md)** - Technical details
- **[Quick Start](client/QUICKSTART.md)** - Get started quickly

## 🌍 Supported Languages

The system supports translation between 200+ languages including:

- English, Spanish, French, German, Italian, Portuguese
- Chinese (Simplified/Traditional), Japanese, Korean
- Arabic, Russian, Hindi, Turkish, Polish
- Dutch, Swedish, Danish, Norwegian, Finnish
- And many more...

## 📖 Usage Examples

### Text-to-Text Translation

```bash
curl -X POST http://localhost:8075/translate \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, how are you?",
    "input_type": "text",
    "source_language": "en",
    "target_language": "es",
    "output_type": "text"
  }'
```

### Using the TypeScript Client

```typescript
import { MultimodalTranslationClient } from '@/lib/api-client';

const client = new MultimodalTranslationClient();

// Translate text
const result = await client.translateText(
  'Hello, world!',
  'en',
  'es'
);
console.log(result); // "¡Hola, mundo!"

// Check service health
const health = await client.healthCheck();
```

## 🔧 Development

### Backend Development

```bash
cd services/asr
uv run python service.py
```

### Frontend Development

```bash
cd client
npm run dev       # Development server
npm run build     # Production build
npm run lint      # Code quality check
```

## 📊 Service Management

```bash
cd services

# Start all services
bash start_all_services.sh

# Check service status
bash check_services.sh

# View logs
bash view_logs.sh

# Stop all services
bash stop_all_services.sh

# Restart specific service
bash restart_service.sh asr
```

## 🐛 Troubleshooting

### Backend Issues

1. **Services won't start**
   ```bash
   # Check if ports are in use
   lsof -i :8075
   lsof -i :8076
   lsof -i :8077
   lsof -i :8078
   ```

2. **Model loading fails**
   - Check model directories exist
   - Ensure sufficient disk space
   - Check logs in `services/logs/`

3. **Service not responding**
   ```bash
   cd services
   bash check_services.sh
   bash restart_service.sh <service_name>
   ```

### Client Issues

1. **Cannot connect to services**
   - Ensure backend services are running
   - Check `.env.local` for correct URLs
   - Verify firewall settings

2. **Microphone not working**
   - Grant browser permissions
   - Use HTTPS or localhost
   - Check browser console for errors

3. **Build errors**
   ```bash
   cd client
   rm -rf .next node_modules
   npm install
   npm run build
   ```

## 🎯 Performance Tips

1. **First request is slower** - Models load on first use
2. **Keep audio short** - < 30 seconds recommended
3. **Use appropriate models** - Base models are faster
4. **Monitor resources** - Check CPU/memory usage
5. **Use API Gateway** - Handles service coordination

## 🔒 Security Considerations

- Use HTTPS in production
- Implement authentication for public deployments
- Configure CORS properly
- Add rate limiting
- Monitor API usage

## 🚀 Deployment

### Docker (Coming Soon)

```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Production Deployment

1. **Backend**: Deploy services on separate servers/containers
2. **Frontend**: Deploy to Vercel, Netlify, or similar
3. **Configure**: Update environment variables
4. **Monitor**: Set up logging and monitoring

## 📝 License

This project uses several open-source technologies:
- Whisper: Apache 2.0
- NLLB: CC-BY-NC
- Coqui XTTS: MPL 2.0
- FastAPI: MIT
- Next.js: MIT

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📬 Support

- **Issues**: Check documentation first
- **Logs**: Review service logs in `services/logs/`
- **Health**: Use health check endpoints
- **Community**: Open GitHub issues for questions

## 🎓 Acknowledgments

- **OpenAI Whisper** - Speech recognition
- **Meta NLLB** - Neural translation
- **Coqui XTTS** - Speech synthesis
- **FastAPI** - Web framework
- **Next.js** - React framework

---

**Built with ❤️ for multilingual communication**
