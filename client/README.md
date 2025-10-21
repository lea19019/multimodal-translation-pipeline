# Multimodal Translation Client

A modern, web-based client application for the Multimodal Translation API. Built with Next.js, TypeScript, and Tailwind CSS, this application provides an intuitive interface for translating text and audio across multiple languages.

## Features

- 🌍 **Text-to-Text Translation**: Translate text between languages
- 🎤 **Audio-to-Text Translation**: Record audio, transcribe, and translate
- 🔊 **Text-to-Audio Translation**: Translate text and generate speech
- 🎵 **Audio-to-Audio Translation**: Complete speech-to-speech translation pipeline
- ✅ **Real-time Health Monitoring**: Check service status in real-time
- 🎨 **Modern UI**: Clean, responsive design with dark mode support
- ⚡ **Fast & Efficient**: Built on Next.js 14 with React Server Components

## Technology Stack

- **Framework**: [Next.js 14](https://nextjs.org/) - React framework with App Router
- **Language**: [TypeScript](https://www.typescriptlang.org/) - Type-safe JavaScript
- **Styling**: [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
- **HTTP Client**: [Axios](https://axios-http.com/) - Promise-based HTTP client
- **Icons**: [Lucide React](https://lucide.dev/) - Beautiful & consistent icons
- **Audio API**: Web Audio API - Native browser audio processing

## Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js**: Version 18.0 or higher ([Download](https://nodejs.org/))
- **npm** or **yarn**: Package manager (comes with Node.js)
- **Translation Services**: The backend services must be running

## Installation

### 1. Clone the Repository

```bash
cd /home/vacl2/multimodal_translation/client
```

### 2. Install Dependencies

Using npm:
```bash
npm install
```

Using yarn:
```bash
yarn install
```

Using pnpm:
```bash
pnpm install
```

### 3. Configure Environment Variables (Optional)

Create a `.env.local` file in the client directory to configure API endpoints:

```bash
# .env.local
NEXT_PUBLIC_GATEWAY_URL=http://localhost:8075
NEXT_PUBLIC_ASR_URL=http://localhost:8076
NEXT_PUBLIC_NMT_URL=http://localhost:8077
NEXT_PUBLIC_TTS_URL=http://localhost:8078
```

If not set, the application uses these default values.

## Running the Application

### Development Mode

Start the development server with hot reload:

```bash
npm run dev
```

The application will be available at **http://localhost:3000**

### Production Build

Build the application for production:

```bash
npm run build
```

Start the production server:

```bash
npm start
```

### Type Checking

Run TypeScript type checking:

```bash
npm run type-check
```

### Linting

Check code quality:

```bash
npm run lint
```

## Project Structure

```
client/
├── src/
│   ├── app/                      # Next.js App Router
│   │   ├── layout.tsx           # Root layout
│   │   ├── page.tsx             # Home page
│   │   └── globals.css          # Global styles
│   ├── components/               # React components
│   │   ├── HealthStatus.tsx     # Service health monitor
│   │   └── TranslationForm.tsx  # Main translation interface
│   ├── hooks/                    # Custom React hooks
│   │   └── useAudioRecorder.ts  # Audio recording hook
│   └── lib/                      # Utilities and libraries
│       └── api-client.ts        # API client library
├── public/                       # Static assets
├── package.json                  # Dependencies
├── tsconfig.json                 # TypeScript config
├── tailwind.config.js            # Tailwind CSS config
├── next.config.js                # Next.js config
└── README.md                     # This file
```

## Usage Guide

### 1. Check Service Health

The application automatically checks the health of all backend services when loaded. You'll see:
- ✅ Green checkmarks for healthy services
- ❌ Red X marks for unavailable services

Click **Refresh** to manually check service status.

### 2. Text-to-Text Translation

1. Select **Text to Text** mode
2. Choose source and target languages
3. Type or paste your text
4. Click **Translate**
5. View the translated text

**Example:**
- Source: English
- Input: "Hello, how are you?"
- Target: Spanish
- Output: "Hola, ¿cómo estás?"

### 3. Audio-to-Text Translation

1. Select **Audio to Text** mode
2. Choose source and target languages
3. Click **Start Recording**
4. Speak into your microphone
5. Click **Stop Recording**
6. Click **Translate**
7. View the translated text

**Note:** You'll need to grant microphone permissions when prompted.

### 4. Text-to-Audio Translation

1. Select **Text to Audio** mode
2. Choose source and target languages
3. Type or paste your text
4. Click **Translate**
5. Click **Play Audio** to hear the translated speech

### 5. Audio-to-Audio Translation

1. Select **Audio to Audio** mode
2. Choose source and target languages
3. Record your audio
4. Click **Translate**
5. Click **Play Audio** to hear the translated speech

## API Client Library

The application includes a comprehensive TypeScript API client that you can use in your own projects:

```typescript
import { MultimodalTranslationClient } from '@/lib/api-client';

// Initialize client
const client = new MultimodalTranslationClient(
  'http://localhost:8075', // Gateway URL
  'http://localhost:8076', // ASR URL
  'http://localhost:8077', // NMT URL
  'http://localhost:8078'  // TTS URL
);

// Text-to-text translation
const result = await client.translateText(
  'Hello, world!',
  'en',
  'es'
);
console.log(result); // "¡Hola, mundo!"

// Check health
const health = await client.healthCheck();
console.log(health.status); // "healthy"

// Get available models
const asrModels = await client.getASRModels();
console.log(asrModels); // ["base", "small", "medium"]
```

### Audio Utilities

The client includes audio utility functions:

```typescript
import { AudioUtils } from '@/lib/api-client';

// Convert Float32Array to base64
const base64Audio = AudioUtils.float32ArrayToBase64(audioData);

// Convert base64 to Float32Array
const audioData = AudioUtils.base64ToFloat32Array(base64Audio);

// Play audio from base64
await AudioUtils.playAudioFromBase64(base64Audio, 22050);
```

## Supported Languages

The application currently supports the following languages:

- 🇬🇧 English (en)
- 🇪🇸 Spanish (es)
- 🇫🇷 French (fr)
- 🇩🇪 German (de)
- 🇮🇹 Italian (it)
- 🇵🇹 Portuguese (pt)
- 🇨🇳 Chinese Simplified (zh)
- 🇯🇵 Japanese (ja)
- 🇰🇷 Korean (ko)
- 🇸🇦 Arabic (ar)
- 🇷🇺 Russian (ru)
- 🇮🇳 Hindi (hi)

More languages can be added by extending the `SUPPORTED_LANGUAGES` array in `src/lib/api-client.ts`.

## Troubleshooting

### Services Not Available

**Problem**: Red error banner showing "Services Unavailable"

**Solution**:
1. Ensure all backend services are running
2. Check service URLs in `.env.local`
3. Verify ports are not blocked by firewall
4. Run `bash services/check_services.sh` in the backend

### Microphone Not Working

**Problem**: Cannot record audio

**Solution**:
1. Grant microphone permissions when prompted
2. Check browser console for errors
3. Ensure you're using HTTPS or localhost (required for microphone access)
4. Try a different browser (Chrome/Edge recommended)

### Audio Playback Issues

**Problem**: Translated audio won't play

**Solution**:
1. Check browser console for errors
2. Ensure TTS service is healthy
3. Try a different browser
4. Check audio settings/volume

### Translation Timeout

**Problem**: Translation takes too long or times out

**Solution**:
1. First translation may take longer (model loading)
2. Check network connection
3. Verify backend services have sufficient resources
4. Try shorter text/audio inputs

### Build Errors

**Problem**: `npm run build` fails

**Solution**:
```bash
# Clear cache and reinstall
rm -rf .next node_modules package-lock.json
npm install
npm run build
```

### Type Errors

**Problem**: TypeScript errors during development

**Solution**:
```bash
# Check types
npm run type-check

# Restart TypeScript server in VS Code
# Command Palette > TypeScript: Restart TS Server
```

## Browser Compatibility

The application works best with modern browsers:

- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ❌ Internet Explorer (not supported)

**Note**: Audio recording requires getUserMedia API support.

## Performance Tips

1. **First Load**: Initial translations may be slower due to model loading
2. **Audio Recording**: Keep recordings under 30 seconds for best performance
3. **Text Length**: Shorter texts translate faster (< 512 tokens recommended)
4. **Network**: Use a stable internet connection for best results
5. **Browser**: Chrome/Edge generally provide best performance

## Security Considerations

- **HTTPS**: Use HTTPS in production for microphone access
- **CORS**: Configure CORS on backend services for production
- **API Keys**: Implement authentication if deploying publicly
- **Rate Limiting**: Consider adding rate limiting for public deployments

## Development

### Adding New Language

Edit `src/lib/api-client.ts`:

```typescript
export const SUPPORTED_LANGUAGES = [
  // ... existing languages
  { code: 'nl', name: 'Dutch', nllb: 'nld_Latn' },
];
```

### Customizing Styles

The application uses Tailwind CSS. Customize colors in `tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      primary: {
        // Your custom colors
      },
    },
  },
}
```

### Adding New Features

1. Create new components in `src/components/`
2. Add new pages in `src/app/`
3. Extend API client in `src/lib/api-client.ts`
4. Add custom hooks in `src/hooks/`

## Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Import project to Vercel
3. Configure environment variables
4. Deploy

### Docker

Create a `Dockerfile`:

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
COPY --from=builder /app/next.config.js ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000
CMD ["npm", "start"]
```

Build and run:

```bash
docker build -t multimodal-translation-client .
docker run -p 3000:3000 multimodal-translation-client
```

### Static Export

For static hosting:

1. Update `next.config.js`:
```javascript
const nextConfig = {
  output: 'export',
}
```

2. Build:
```bash
npm run build
```

3. Deploy the `out/` directory

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the Multimodal Translation System and follows the same licensing as the backend services.

## Support

For issues and questions:

1. Check this README
2. Review API Documentation (`services/API_DOCUMENTATION.md`)
3. Check backend service logs
4. Open an issue on GitHub

## Acknowledgments

- **Next.js** - The React Framework
- **Tailwind CSS** - Utility-first CSS
- **Lucide** - Beautiful icons
- **OpenAI Whisper** - Speech recognition
- **Meta NLLB** - Neural translation
- **Coqui XTTS** - Speech synthesis

---

**Built with ❤️ using Next.js and TypeScript**
