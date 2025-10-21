# Multimodal Translation Web Client

## Overview

This is a modern web application built with **Next.js 14** and **TypeScript** that provides a user-friendly interface for the Multimodal Translation API. The client enables users to translate text and audio across multiple languages using state-of-the-art AI models.

## Quick Start

### Prerequisites
- Node.js 18+ installed
- Backend services running (see `services/README.md`)

### Installation & Running

```bash
# Navigate to client directory
cd client

# Run the quick start script
bash start.sh
```

Or manually:

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at **http://localhost:3000**

## Architecture

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | Next.js 14 (App Router) | Server-side rendering, routing, optimization |
| **Language** | TypeScript | Type safety and developer experience |
| **Styling** | Tailwind CSS | Utility-first responsive design |
| **State** | React Hooks | Component state management |
| **HTTP** | Axios | API communication with retry logic |
| **Audio** | Web Audio API | Audio recording and playback |
| **Icons** | Lucide React | Consistent, modern iconography |

### Project Structure

```
client/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── layout.tsx         # Root layout with metadata
│   │   ├── page.tsx           # Home page with tab navigation
│   │   └── globals.css        # Global styles and Tailwind imports
│   │
│   ├── components/             # Reusable React components
│   │   ├── HealthStatus.tsx   # Service health monitoring
│   │   └── TranslationForm.tsx # Main translation interface
│   │
│   ├── hooks/                  # Custom React hooks
│   │   └── useAudioRecorder.ts # Audio recording with MediaRecorder API
│   │
│   └── lib/                    # Utility libraries
│       └── api-client.ts      # Comprehensive API client with types
│
├── public/                     # Static assets
├── package.json               # Dependencies and scripts
├── tsconfig.json              # TypeScript configuration
├── tailwind.config.js         # Tailwind CSS configuration
├── next.config.js             # Next.js configuration
├── .env.example               # Environment variables template
├── start.sh                   # Quick start script
└── README.md                  # Comprehensive documentation
```

## Key Features

### 1. Translation Modes

The client supports four translation modes accessible via tabs:

#### Text to Text
- Input: Text area for source text
- Output: Translated text
- Use case: Document translation, messaging

#### Audio to Text
- Input: Microphone recording
- Output: Transcribed and translated text
- Use case: Voice memos, interviews

#### Text to Audio
- Input: Text area for source text
- Output: Audio player with synthesized speech
- Use case: Language learning, accessibility

#### Audio to Audio
- Input: Microphone recording
- Output: Audio player with translated speech
- Use case: Real-time conversation translation

### 2. Real-time Health Monitoring

- Automatic health checks every 30 seconds
- Visual indicators for each service (ASR, NMT, TTS)
- Manual refresh capability
- Error handling and retry logic

### 3. Audio Recording

- Browser-native MediaRecorder API
- Real-time recording feedback
- Automatic resampling to 16kHz for ASR
- Clear and re-record functionality

### 4. Audio Playback

- Web Audio API for high-quality playback
- Visual feedback during playback
- Support for various sample rates

### 5. Error Handling

- User-friendly error messages
- Service-specific error reporting
- Timeout handling
- Network error recovery

## API Client Library

The application includes a comprehensive, reusable TypeScript API client (`src/lib/api-client.ts`) that can be used independently:

### Features

- ✅ Full TypeScript types for all API endpoints
- ✅ Separate methods for each translation mode
- ✅ Direct service access (bypassing gateway)
- ✅ Health check and model listing
- ✅ Audio utility functions
- ✅ Comprehensive error handling
- ✅ Configurable timeouts

### Example Usage

```typescript
import { MultimodalTranslationClient, AudioUtils } from '@/lib/api-client';

// Initialize client
const client = new MultimodalTranslationClient();

// Text-to-text translation
const translated = await client.translateText(
  'Hello, world!',
  'en',
  'es'
);

// Audio-to-text translation
const audioBase64 = AudioUtils.float32ArrayToBase64(audioData);
const transcribed = await client.translateAudioToText(
  audioBase64,
  'en',
  'es'
);

// Check health
const health = await client.healthCheck();

// Get available models
const models = await client.getASRModels();
```

## Component Details

### HealthStatus Component

**File**: `src/components/HealthStatus.tsx`

**Purpose**: Displays real-time status of backend services

**Features**:
- Auto-refresh every 30 seconds
- Visual indicators (green/red)
- Service-specific information
- Manual refresh button
- Error state handling

**State**:
- `health`: Current health data from API
- `loading`: Loading indicator
- `error`: Error message if check fails

### TranslationForm Component

**File**: `src/components/TranslationForm.tsx`

**Purpose**: Main translation interface with mode-specific UI

**Props**:
- `mode`: Translation mode (text | audio-to-text | text-to-audio | audio-to-audio)

**Features**:
- Dynamic UI based on mode
- Language selection dropdowns
- Text input or audio recording
- Translation button with loading state
- Output display (text or audio player)
- Comprehensive error handling

**State**:
- `inputText`: Source text input
- `outputText`: Translated text output
- `outputAudio`: Base64 encoded audio output
- `sourceLanguage`: Selected source language
- `targetLanguage`: Selected target language
- `loading`: Translation in progress
- `error`: Error message
- `isPlayingAudio`: Audio playback state

### useAudioRecorder Hook

**File**: `src/hooks/useAudioRecorder.ts`

**Purpose**: Custom hook for audio recording functionality

**Returns**:
- `isRecording`: Recording state boolean
- `audioData`: Float32Array of recorded audio
- `audioBlob`: Blob of recorded audio
- `startRecording()`: Start recording function
- `stopRecording()`: Stop recording function
- `clearRecording()`: Clear recorded audio
- `error`: Error message if recording fails

**Implementation**:
- Uses MediaRecorder API
- Automatically resamples to 16kHz
- Converts to Float32Array for ASR
- Handles microphone permissions

## Styling

### Tailwind CSS Configuration

The application uses a custom Tailwind configuration with:

- **Primary Colors**: Blue/Indigo palette
- **Dark Mode**: Full dark mode support
- **Responsive**: Mobile-first responsive design
- **Custom Utilities**: Extended with project-specific utilities

### Design Principles

1. **Clean & Modern**: Minimalist interface with focus on functionality
2. **Responsive**: Works on mobile, tablet, and desktop
3. **Accessible**: WCAG-compliant color contrasts
4. **Consistent**: Unified spacing, sizing, and colors
5. **Dark Mode**: Automatic dark mode support

## Development Guide

### Running Development Server

```bash
npm run dev
```

Runs on http://localhost:3000 with:
- Hot reload for instant feedback
- Fast Refresh for preserving state
- Detailed error overlays

### Building for Production

```bash
npm run build
npm start
```

Optimizations:
- Code splitting
- Image optimization
- Static generation where possible
- Minification and compression

### Type Checking

```bash
npm run type-check
```

Runs TypeScript compiler without emitting files to check for type errors.

### Linting

```bash
npm run lint
```

Runs ESLint with Next.js recommended rules.

## Configuration

### Environment Variables

Create `.env.local` from `.env.example`:

```bash
NEXT_PUBLIC_GATEWAY_URL=http://localhost:8075
NEXT_PUBLIC_ASR_URL=http://localhost:8076
NEXT_PUBLIC_NMT_URL=http://localhost:8077
NEXT_PUBLIC_TTS_URL=http://localhost:8078
```

### Next.js Config

**File**: `next.config.js`

```javascript
const nextConfig = {
  reactStrictMode: true,  // Enable React strict mode
  swcMinify: true,        // Use SWC for minification
}
```

### TypeScript Config

**File**: `tsconfig.json`

Configured for:
- ES2020 target
- Strict mode enabled
- Path aliases (`@/*` → `src/*`)
- Next.js optimizations

## Browser Support

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 90+ | ✅ Fully supported |
| Edge | 90+ | ✅ Fully supported |
| Firefox | 88+ | ✅ Fully supported |
| Safari | 14+ | ✅ Supported |
| IE | Any | ❌ Not supported |

**Requirements**:
- JavaScript enabled
- getUserMedia API (for audio recording)
- Web Audio API (for playback)
- ES2020 support

## Performance Considerations

### Optimizations

1. **Code Splitting**: Automatic route-based splitting
2. **Image Optimization**: Next.js Image component
3. **Font Optimization**: Google Fonts optimization
4. **Bundle Analysis**: `npm run build` shows bundle sizes

### Best Practices

1. **Keep recordings < 30 seconds** for faster processing
2. **Use shorter texts** (< 512 tokens) for faster translation
3. **First translation is slower** due to model loading
4. **Cache translations** on client side if needed

## Deployment

### Vercel (Recommended)

1. Push to GitHub
2. Import to Vercel
3. Configure environment variables
4. Deploy automatically

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Static Export

For static hosting (limited functionality):

```javascript
// next.config.js
const nextConfig = {
  output: 'export',
}
```

## Troubleshooting

### Common Issues

1. **Services not available**: Start backend services first
2. **Microphone not working**: Grant permissions, use HTTPS/localhost
3. **Build errors**: Clear `.next` and `node_modules`, reinstall
4. **Type errors**: Run `npm run type-check` for details

### Debug Mode

Enable verbose logging:

```javascript
// In browser console
localStorage.setItem('debug', '*')
```

## Testing

Currently, the application doesn't have automated tests, but you can add:

```bash
npm install -D @testing-library/react @testing-library/jest-dom jest
```

## Contributing

To contribute:

1. Fork the repository
2. Create a feature branch
3. Make changes with proper TypeScript types
4. Test thoroughly
5. Submit a pull request

## License

Part of the Multimodal Translation System project.

## Support

- **Documentation**: See README.md in this directory
- **API Docs**: See `services/API_DOCUMENTATION.md`
- **Issues**: Check backend service logs first
- **Questions**: Review this document and API documentation

---

**Built with Next.js 14, TypeScript, and Tailwind CSS**
