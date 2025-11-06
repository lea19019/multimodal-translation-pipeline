import axios, { AxiosInstance, AxiosError } from 'axios';

/**
 * Type definitions for the Multimodal Translation API
 */
export type InputType = 'text' | 'audio';
export type OutputType = 'text' | 'audio';

export interface TranslationRequest {
  input: string;
  input_type: InputType;
  source_language: string;
  target_language: string;
  output_type: OutputType;
  asr_model?: string;
  nmt_model?: string;
  tts_model?: string;
  save_for_evaluation?: boolean;
}

export interface TranslationResponse {
  output: string;
  output_type: OutputType;
  source_language: string;
  target_language: string;
  // Intermediate results from pipeline steps
  transcribed_text?: string | null;  // From ASR (if audio input)
  translated_text?: string | null;   // From NMT (always available)
  output_audio?: string | null;      // From TTS (if audio output)
}

export interface HealthResponse {
  status: string;
  service: string;
  downstream_services?: {
    asr?: ServiceHealth;
    nmt?: ServiceHealth;
    tts?: ServiceHealth;
  };
}

export interface ServiceHealth {
  status: string;
  service: string;
  device?: string;
}

export interface ModelsResponse {
  models: string[];
}

export interface TranscribeRequest {
  audio: string;
  source_language?: string;
  model_name?: string;
}

export interface TranscribeResponse {
  text: string;
  language: string;
}

export interface NMTTranslateRequest {
  text: string;
  source_language: string;
  target_language: string;
  model_name?: string;
}

export interface NMTTranslateResponse {
  translated_text: string;
  source_language: string;
  target_language: string;
}

export interface SynthesizeRequest {
  text: string;
  language: string;
  model_name?: string;
}

export interface SynthesizeResponse {
  audio: string;
}

export interface APIError {
  detail: string;
}

/**
 * Client for the Multimodal Translation API
 * Provides methods to interact with the translation services
 */
export class MultimodalTranslationClient {
  private gatewayClient: AxiosInstance;
  private asrClient: AxiosInstance;
  private nmtClient: AxiosInstance;
  private ttsClient: AxiosInstance;

  constructor(
    gatewayUrl: string = 'http://localhost:8075',
    asrUrl: string = 'http://localhost:8076',
    nmtUrl: string = 'http://localhost:8077',
    ttsUrl: string = 'http://localhost:8078'
  ) {
    // Create axios instances for each service
    this.gatewayClient = axios.create({
      baseURL: gatewayUrl,
      timeout: 120000, // 2 minutes for full pipeline
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.asrClient = axios.create({
      baseURL: asrUrl,
      timeout: 60000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.nmtClient = axios.create({
      baseURL: nmtUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.ttsClient = axios.create({
      baseURL: ttsUrl,
      timeout: 60000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Main translation method - handles all translation pipelines
   */
  async translate(request: TranslationRequest): Promise<TranslationResponse> {
    try {
      const response = await this.gatewayClient.post<TranslationResponse>(
        '/translate',
        request
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Translate text to text
   */
  async translateText(
    text: string,
    sourceLanguage: string,
    targetLanguage: string,
    nmtModel: string = 'base'
  ): Promise<string> {
    const result = await this.translate({
      input: text,
      input_type: 'text',
      source_language: sourceLanguage,
      target_language: targetLanguage,
      output_type: 'text',
      nmt_model: nmtModel,
    });
    return result.output;
  }

  /**
   * Transcribe and translate audio to text
   */
  async translateAudioToText(
    audioBase64: string,
    sourceLanguage: string,
    targetLanguage: string,
    asrModel: string = 'base',
    nmtModel: string = 'base'
  ): Promise<string> {
    const result = await this.translate({
      input: audioBase64,
      input_type: 'audio',
      source_language: sourceLanguage,
      target_language: targetLanguage,
      output_type: 'text',
      asr_model: asrModel,
      nmt_model: nmtModel,
    });
    return result.output;
  }

  /**
   * Translate text and synthesize speech
   */
  async translateTextToAudio(
    text: string,
    sourceLanguage: string,
    targetLanguage: string,
    nmtModel: string = 'base',
    ttsModel: string = 'base'
  ): Promise<string> {
    const result = await this.translate({
      input: text,
      input_type: 'text',
      source_language: sourceLanguage,
      target_language: targetLanguage,
      output_type: 'audio',
      nmt_model: nmtModel,
      tts_model: ttsModel,
    });
    return result.output;
  }

  /**
   * Full pipeline: transcribe, translate, and synthesize
   */
  async translateAudioToAudio(
    audioBase64: string,
    sourceLanguage: string,
    targetLanguage: string,
    asrModel: string = 'base',
    nmtModel: string = 'base',
    ttsModel: string = 'base'
  ): Promise<string> {
    const result = await this.translate({
      input: audioBase64,
      input_type: 'audio',
      source_language: sourceLanguage,
      target_language: targetLanguage,
      output_type: 'audio',
      asr_model: asrModel,
      nmt_model: nmtModel,
      tts_model: ttsModel,
    });
    return result.output;
  }

  /**
   * Check health of all services via API Gateway
   */
  async healthCheck(): Promise<HealthResponse> {
    try {
      const response = await this.gatewayClient.get<HealthResponse>('/health');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get available ASR models (from individual service)
   */
  async getASRModels(): Promise<string[]> {
    try {
      const response = await this.asrClient.get<ModelsResponse>('/models');
      return response.data.models;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get available ASR models (from gateway)
   */
  async getASRModelsFromGateway(): Promise<string[]> {
    try {
      const response = await this.gatewayClient.get<ModelsResponse>('/models/asr');
      return response.data.models;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get available NMT models (from individual service)
   */
  async getNMTModels(): Promise<string[]> {
    try {
      const response = await this.nmtClient.get<ModelsResponse>('/models');
      return response.data.models;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get available NMT models (from gateway)
   */
  async getNMTModelsFromGateway(): Promise<string[]> {
    try {
      const response = await this.gatewayClient.get<ModelsResponse>('/models/nmt');
      return response.data.models;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get available TTS models (from individual service)
   */
  async getTTSModels(): Promise<string[]> {
    try {
      const response = await this.ttsClient.get<ModelsResponse>('/models');
      return response.data.models;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get available TTS models (from gateway)
   */
  async getTTSModelsFromGateway(): Promise<string[]> {
    try {
      const response = await this.gatewayClient.get<ModelsResponse>('/models/tts');
      return response.data.models;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Direct ASR transcription (bypasses API Gateway)
   */
  async transcribe(
    audioBase64: string,
    sourceLanguage?: string,
    modelName?: string
  ): Promise<TranscribeResponse> {
    try {
      const response = await this.asrClient.post<TranscribeResponse>(
        '/transcribe',
        {
          audio: audioBase64,
          source_language: sourceLanguage,
          model_name: modelName,
        } as TranscribeRequest
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Direct NMT translation (bypasses API Gateway)
   */
  async nmtTranslate(
    text: string,
    sourceLanguage: string,
    targetLanguage: string,
    modelName?: string
  ): Promise<NMTTranslateResponse> {
    try {
      const response = await this.nmtClient.post<NMTTranslateResponse>(
        '/translate',
        {
          text,
          source_language: sourceLanguage,
          target_language: targetLanguage,
          model_name: modelName,
        } as NMTTranslateRequest
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Direct TTS synthesis (bypasses API Gateway)
   */
  async synthesize(
    text: string,
    language: string,
    modelName?: string
  ): Promise<SynthesizeResponse> {
    try {
      const response = await this.ttsClient.post<SynthesizeResponse>(
        '/synthesize',
        {
          text,
          language,
          model_name: modelName,
        } as SynthesizeRequest
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Error handler for API calls
   */
  private handleError(error: unknown): Error {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<APIError>;
      if (axiosError.response?.data?.detail) {
        return new Error(axiosError.response.data.detail);
      }
      if (axiosError.message) {
        return new Error(axiosError.message);
      }
    }
    if (error instanceof Error) {
      return error;
    }
    return new Error('An unknown error occurred');
  }
}

/**
 * Audio utility functions
 */
export class AudioUtils {
  /**
   * Convert Float32Array to base64 string
   */
  static float32ArrayToBase64(audioData: Float32Array): string {
    const buffer = audioData.buffer;
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  /**
   * Convert base64 string to Float32Array
   */
  static base64ToFloat32Array(base64: string): Float32Array {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
  }

  /**
   * Resample audio to target sample rate
   */
  static async resampleAudio(
    audioBuffer: AudioBuffer,
    targetSampleRate: number
  ): Promise<Float32Array> {
    const offlineContext = new OfflineAudioContext(
      1,
      audioBuffer.duration * targetSampleRate,
      targetSampleRate
    );

    const source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineContext.destination);
    source.start();

    const resampledBuffer = await offlineContext.startRendering();
    return resampledBuffer.getChannelData(0);
  }

  /**
   * Create AudioBuffer from Float32Array
   */
  static createAudioBuffer(
    audioContext: AudioContext,
    audioData: Float32Array,
    sampleRate: number
  ): AudioBuffer {
    const buffer = audioContext.createBuffer(1, audioData.length, sampleRate);
    buffer.copyToChannel(audioData, 0);
    return buffer;
  }

  /**
   * Play audio from base64 string
   */
  static async playAudioFromBase64(
    base64Audio: string,
    sampleRate: number = 22050
  ): Promise<void> {
    const audioContext = new AudioContext();
    const audioData = this.base64ToFloat32Array(base64Audio);
    const audioBuffer = this.createAudioBuffer(audioContext, audioData, sampleRate);

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start();

    return new Promise((resolve) => {
      source.onended = () => {
        audioContext.close();
        resolve();
      };
    });
  }
}

/**
 * Language utilities
 */
export const SUPPORTED_LANGUAGES = [
  { code: 'en', name: 'English', nllb: 'eng_Latn', tts: 'en' },
  { code: 'es', name: 'Spanish', nllb: 'spa_Latn', tts: 'es' },
  { code: 'efi', name: 'Efik', nllb: 'efi_Latn', tts: 'efi' },
  { code: 'ibo', name: 'Igbo', nllb: 'ibo_Latn', tts: 'ibo' },
  { code: 'xho', name: 'Xhosa', nllb: 'xho_Latn', tts: 'xho' },
  { code: 'swa', name: 'Swahili', nllb: 'swh_Latn', tts: 'swa' },
];

export function getLanguageName(code: string): string {
  const language = SUPPORTED_LANGUAGES.find((lang) => lang.code === code);
  return language?.name || code;
}
