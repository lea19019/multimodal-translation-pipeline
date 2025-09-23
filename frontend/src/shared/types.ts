// Translation Pipeline Types
export interface TranslationRequest {
  id: string;
  mode: 'evaluation' | 'free';
  type: 'text-to-text' | 'text-to-speech' | 'speech-to-text' | 'speech-to-speech';
  sourceLang: string;
  targetLang: string;
  input: string | File;
  reference?: string | File; // For evaluation mode
  pipelineConfig?: PipelineConfig; // Legacy field
  pipelineId?: string; // New field for Model Manager integration
  options?: any; // Additional processing options
}

export interface TranslationResponse {
  id?: string;
  success?: boolean; // For Model Manager compatibility
  result?: any; // Flexible result field
  output?: string | Blob; // Legacy field
  evaluation?: EvaluationMetrics; // Renamed from evaluationMetrics
  evaluationMetrics?: EvaluationMetrics; // Legacy field
  metadata?: any; // Processing metadata
  processingTime?: number;
  status?: 'success' | 'error';
  error?: string | any; // Can be string or error object
  timestamp?: string;
}

export interface PipelineConfig {
  id: string;
  name: string;
  description: string;
  models: {
    asr?: ModelInfo;
    nmt?: ModelInfo;
    tts?: ModelInfo;
  };
}

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  type: 'asr' | 'nmt' | 'tts';
  language?: string;
  description?: string;
}

export interface EvaluationMetrics {
  bleu?: number;
  comet?: number;
  wer?: number; // Word Error Rate for ASR
  mos?: number; // Mean Opinion Score for TTS
  latency?: number;
  throughput?: number;
}

export interface BatchTranslationRequest {
  requests: TranslationRequest[];
  batchId: string;
}

export interface BatchTranslationResponse {
  batchId: string;
  results: TranslationResponse[];
  overallMetrics?: {
    averageBleu?: number;
    averageComet?: number;
    averageLatency?: number;
    successRate?: number;
  };
}

// File Upload Types
export interface FileUpload {
  file: File;
  type: 'text' | 'audio' | 'json';
  format: string;
}

// API Response wrapper
export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
}