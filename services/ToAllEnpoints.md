Here is API documentation for a model service (ASR, TTS, or NMT) that ToAll can call. This describes the REST API endpoints your model should expose so that ToAll (or its service adapter) can interact with it.

---

# Model Service API Specification (ASR, TTS, NMT)

## 1. ASR (Automatic Speech Recognition) Service

### Endpoint: Transcribe Audio

- **URL:** `/asr/transcribe`
- **Method:** `POST`
- **Request:**
  - `Content-Type: multipart/form-data`
  - Fields:
    - `audio`: Audio file (e.g., WAV, MP3)
    - `language` (optional): Language code (e.g., "en", "es")
    - `options` (optional): JSON string with options (e.g., `{ "diarization": true }`)
- **Response:**
  - `200 OK`
  - `application/json`
  - Example:
    ```json
    {
      "segments": [
        { "start": 0.0, "end": 2.5, "text": "Hello world", "confidence": 0.98 },
        { "start": 2.6, "end": 5.0, "text": "How are you?", "confidence": 0.95 }
      ],
      "text": "Hello world How are you?"
    }
    ```

---

## 2. TTS (Text-to-Speech) Service

### Endpoint: Synthesize Speech

- **URL:** `/tts/synthesize`
- **Method:** `POST`
- **Request:**
  - `Content-Type: application/json`
  - Body:
    ```json
    {
      "text": "Hello world",
      "language": "en",
      "voice": "en-US-Standard-A",
      "options": { "speed": 1.0, "pitch": 0.0 }
    }
    ```
- **Response:**
  - `200 OK`
  - `audio/wav` (or `audio/mp3`)
  - Audio file in response body

---

## 3. NMT (Neural Machine Translation) Service

### Endpoint: Translate Text

- **URL:** `/nmt/translate`
- **Method:** `POST`
- **Request:**
  - `Content-Type: application/json`
  - Body:
    ```json
    {
      "text": "Hello world",
      "sourceLanguage": "en",
      "targetLanguage": "es",
      "options": { "formality": "informal" }
    }
    ```
- **Response:**
  - `200 OK`
  - `application/json`
  - Example:
    ```json
    {
      "translatedText": "Hola mundo",
      "detectedSourceLanguage": "en"
    }
    ```

---

## General Notes

- All endpoints should return appropriate error codes and messages for invalid input or server errors.
- For ASR, segment timing is in seconds, and confidence is optional.
- For TTS, the response is raw audio data.
- For NMT, the response is translated text and optionally the detected source language.

---

This API spec allows ToAll (or its service adapter) to call your model service in a standardized way. If you need a different protocol (e.g., gRPC), let me know!