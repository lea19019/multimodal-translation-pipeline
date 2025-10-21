# 🚀 Quick Start Guide

## What You Have

I've created a complete **TypeScript web client** for your Multimodal Translation API using:

- **Next.js 14** - Modern React framework
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Beautiful, responsive UI
- **Web Audio API** - Audio recording and playback

## File Structure

```
client/
├── src/
│   ├── app/                    # Pages and layouts
│   ├── components/             # React components
│   ├── hooks/                  # Custom hooks
│   └── lib/                    # API client library
├── package.json                # Dependencies
├── README.md                   # Full documentation
├── ARCHITECTURE.md             # Technical details
└── start.sh                    # Quick start script
```

## How to Run

### Option 1: Quick Start Script (Recommended)

```bash
cd client
bash start.sh
```

This script will:
1. ✅ Check Node.js installation
2. 📦 Install dependencies
3. 🔍 Check backend services
4. 🚀 Start development server

### Option 2: Manual Setup

```bash
cd client
npm install
npm run dev
```

Then open http://localhost:3000

## Features

### 🌍 Four Translation Modes

1. **Text to Text** - Translate written text
2. **Audio to Text** - Record speech, get translated text
3. **Text to Audio** - Translate text, hear it spoken
4. **Audio to Audio** - Full speech-to-speech translation

### ✨ Key Capabilities

- ✅ Real-time service health monitoring
- 🎤 Browser-based audio recording
- 🔊 Audio playback of translations
- 🌓 Dark mode support
- 📱 Mobile responsive
- ⚡ Fast and efficient

### 🌎 Supported Languages

English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, Russian, Hindi (and more!)

## Important Notes

### Before Starting

1. **Backend services must be running** on ports 8075-8078
2. Start them with:
   ```bash
   cd services
   bash start_all_services.sh
   ```

### Browser Requirements

- Chrome/Edge 90+ (recommended)
- Firefox 88+
- Safari 14+
- Microphone permissions needed for audio recording

### First Run

- Initial translations may be slower (model loading)
- Grant microphone permission when prompted
- Check service health status in the UI

## Using the API Client

The client includes a reusable TypeScript API library:

```typescript
import { MultimodalTranslationClient } from '@/lib/api-client';

const client = new MultimodalTranslationClient();

// Translate text
const result = await client.translateText(
  'Hello!',
  'en',
  'es'
);
```

See `src/lib/api-client.ts` for full API.

## Project Commands

```bash
npm run dev         # Start development server
npm run build       # Build for production
npm start           # Run production build
npm run lint        # Check code quality
npm run type-check  # Check TypeScript types
```

## Documentation

- **README.md** - Complete user guide and setup instructions
- **ARCHITECTURE.md** - Technical architecture and component details
- **API_DOCUMENTATION.md** - Backend API reference (in services/)

## Troubleshooting

### Services Not Available
→ Start backend services first (see above)

### Microphone Not Working
→ Grant permissions, ensure HTTPS or localhost

### Build Errors
→ Delete `.next/` and `node_modules/`, reinstall

### Translation Timeout
→ First translation is slower, be patient

## Next Steps

1. **Start backend services** (if not already running)
2. **Run the client** with `bash start.sh`
3. **Open browser** to http://localhost:3000
4. **Try translating** text or audio!

## Need Help?

- Check the README.md for detailed information
- Review ARCHITECTURE.md for technical details
- Check backend logs in `services/logs/`
- Ensure all services are healthy (green checkmarks in UI)

---

**Enjoy your multimodal translation client!** 🎉
