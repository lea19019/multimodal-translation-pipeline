# Multimodal Translation Pipeline

A comprehensive web application for multilingual translation supporting multiple modalities: text-to-text, speech-to-text, text-to-speech, and speech-to-speech translation.

## 🌟 Features

- **Multi-modal Translation**: Support for text, audio, and file inputs
- **Real-time Processing**: Live audio recording and instant translation
- **Multiple Language Support**: 10+ languages including English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, and Korean
- **Flexible Pipeline Configuration**: Customizable translation models and settings
- **Batch Processing**: Handle multiple translation requests simultaneously
- **Comprehensive API**: RESTful API with full OpenAPI documentation
- **Modern UI**: Responsive React interface with real-time feedback

## 🏗️ Architecture

The system consists of three main components:

### 1. Frontend (React + TypeScript)
- **Location**: `frontend/`
- **Port**: 5173
- **Technology**: React 18, TypeScript, Vite, Tailwind CSS
- **Features**: File upload, audio recording, real-time translation interface

### 2. API Gateway (Node.js + Express)
- **Location**: `frontend/src/server/`
- **Port**: 3001
- **Technology**: Express.js, TypeScript, Multer
- **Features**: Request routing, file handling, integration with Model Manager

### 3. Model Manager (Python + FastAPI)
- **Location**: `model-manager/`
- **Port**: 8000
- **Technology**: FastAPI, Python 3.9+, Uvicorn
- **Features**: Translation processing, model management, batch operations

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+
- Git

### Option 1: One-Command Startup
```bash
# Clone and start everything
git clone <repository-url>
cd multimodal-translation-pipeline
./start.sh
```

### Option 2: Manual Setup

1. **Setup Model Manager**:
```bash
cd model-manager
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

2. **Setup Frontend & API Gateway**:
```bash
cd frontend
npm install
npm run dev:server  # Start API Gateway (port 3001)
npm run dev         # Start Frontend (port 5173)
```

3. **Access the Application**:
- Frontend: http://localhost:5173
- API Gateway: http://localhost:3001
- Model Manager: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🧪 Testing

### Test Model Manager API
```bash
cd model-manager
python test_api.py
```

### Test Frontend Integration
1. Open http://localhost:5173
2. Try text translation
3. Test audio recording
4. Upload files for processing

## 📊 Supported Translation Types

| Type | Input | Output | Description |
|------|-------|--------|-------------|
| **Text-to-Text** | Text | Text | Traditional text translation |
| **Speech-to-Text** | Audio | Text | Transcribe and translate speech |
| **Text-to-Speech** | Text | Audio | Translate text and synthesize speech |
| **Speech-to-Speech** | Audio | Audio | Full speech translation pipeline |

## 🌍 Supported Languages

- **English** (en)
- **Spanish** (es) 
- **French** (fr)
- **German** (de)
- **Italian** (it)
- **Portuguese** (pt)
- **Russian** (ru)
- **Chinese** (zh)
- **Japanese** (ja)
- **Korean** (ko)

## 📁 Project Structure

```
multimodal-translation-pipeline/
├── frontend/                      # React application
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── server/              # API Gateway (Express.js)
│   │   │   ├── index.ts         # Main server file
│   │   │   ├── model-manager-client.ts  # Python Model Manager client
│   │   │   └── translation-service.ts   # Translation service wrapper
│   │   ├── types/               # TypeScript type definitions
│   │   └── utils/               # Utility functions
│   ├── package.json
│   └── vite.config.ts
├── model-manager/                 # Python FastAPI service
│   ├── main.py                   # FastAPI application
│   ├── models.py                 # Pydantic models
│   ├── config.py                 # Configuration
│   ├── run.py                    # Development server
│   ├── test_api.py              # API test suite
│   ├── requirements.txt          # Python dependencies
│   └── Dockerfile               # Container configuration
├── docs/                         # Documentation
│   ├── API_DOCUMENTATION.md     # API specifications
│   └── MODEL_MANAGER_INTEGRATION.md  # Integration guide
├── start.sh                      # One-command startup script
└── README.md                     # This file
```

## 🔌 API Endpoints

### Model Manager (Port 8000)
- `GET /health` - Health check
- `GET /pipelines` - List available pipelines
- `POST /translate/text` - Text-to-text translation
- `POST /transcribe` - Speech-to-text transcription
- `POST /synthesize` - Text-to-speech synthesis
- `POST /batch/process` - Batch processing

### API Gateway (Port 3001)
- `GET /api/health` - Health check
- `GET /api/pipelines` - Pipeline configurations
- `POST /api/translate` - Universal translation endpoint
- `POST /api/batch` - Batch translation

## 🛠️ Development

### Adding New Languages
1. Update language lists in `frontend/src/types/index.ts`
2. Add language support in `model-manager/main.py`
3. Update pipeline configurations

### Adding New Models
1. Add model definitions in `model-manager/models.py`
2. Update pipeline configurations
3. Implement model-specific processing logic

### Environment Variables
```bash
# Model Manager
MODEL_MANAGER_HOST=localhost
MODEL_MANAGER_PORT=8000

# API Gateway
API_GATEWAY_PORT=3001
FRONTEND_PORT=5173
```

## 🐳 Docker Support

Build and run with Docker:
```bash
cd model-manager
docker build -t translation-model-manager .
docker run -p 8000:8000 translation-model-manager
```

## 📚 Documentation

- **API Documentation**: `docs/API_DOCUMENTATION.md`
- **Integration Guide**: `docs/MODEL_MANAGER_INTEGRATION.md`
- **Interactive API Docs**: http://localhost:8000/docs (when running)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔧 Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   pkill -f "python run.py"
   pkill -f "node.*3001"
   ```

2. **Python dependencies**:
   ```bash
   cd model-manager
   pip install -r requirements.txt
   ```

3. **Node.js dependencies**:
   ```bash
   cd frontend
   npm install
   ```

4. **CORS issues**: Ensure API Gateway is running on port 3001

### Debug Mode
- Model Manager: Set `DEBUG=True` in `config.py`
- Frontend: Check browser console for errors
- API Gateway: Check terminal logs

## 🎯 Roadmap

- [ ] Real model integration (Whisper, OPUS-MT, eSpeak)
- [ ] User authentication and sessions
- [ ] Translation history and favorites
- [ ] Advanced pipeline configuration UI
- [ ] Performance monitoring and analytics
- [ ] Multi-tenant support
- [ ] Cloud deployment guides