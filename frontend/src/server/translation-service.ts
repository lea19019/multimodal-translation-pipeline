import { ModelManagerClient, ModelManagerError } from './model-manager-client.js';
import { TranslationRequest, TranslationResponse, PipelineConfig } from '../shared/types.js';

class TranslationService {
  private modelManager: ModelManagerClient;

  constructor() {
    this.modelManager = new ModelManagerClient();
  }

  async healthCheck() {
    try {
      return await this.modelManager.healthCheck();
    } catch (error) {
      console.error('Model Manager health check failed:', error);
      return { status: 'unhealthy', error: error.message };
    }
  }

  async getPipelines(): Promise<PipelineConfig[]> {
    try {
      const pipelines = await this.modelManager.getPipelines();
      return pipelines.map(this.convertPipelineFormat);
    } catch (error) {
      console.error('Failed to get pipelines from Model Manager:', error);
      // Fallback to mock data if Model Manager is unavailable
      return this.getMockPipelines();
    }
  }

  async getPipeline(pipelineId: string): Promise<PipelineConfig | null> {
    try {
      const pipeline = await this.modelManager.getPipeline(pipelineId);
      return this.convertPipelineFormat(pipeline);
    } catch (error) {
      console.error(`Failed to get pipeline ${pipelineId} from Model Manager:`, error);
      // Fallback to mock data
      const mockPipelines = this.getMockPipelines();
      return mockPipelines.find(p => p.id === pipelineId) || null;
    }
  }

  async processTranslation(request: TranslationRequest): Promise<TranslationResponse> {
    try {
      const startTime = Date.now();
      
      // Get pipeline configuration
      const pipeline = await this.getPipeline(request.pipelineId);
      if (!pipeline) {
        throw new Error(`Pipeline '${request.pipelineId}' not found`);
      }

      let result: any;
      
      switch (request.type) {
        case 'text-to-text':
          result = await this.handleTextToText(request, pipeline);
          break;
        case 'text-to-speech':
          result = await this.handleTextToSpeech(request, pipeline);
          break;
        case 'speech-to-text':
          result = await this.handleSpeechToText(request, pipeline);
          break;
        case 'speech-to-speech':
          result = await this.handleSpeechToSpeech(request, pipeline);
          break;
        default:
          throw new Error(`Unsupported translation type: ${request.type}`);
      }

      const processingTime = Date.now() - startTime;

      return {
        success: true,
        result: result.result || result,
        evaluation: request.mode === 'evaluation' ? this.generateEvaluation(request, result) : undefined,
        metadata: {
          processingTime,
          pipelineId: request.pipelineId,
          type: request.type,
          ...(result.metadata || {})
        },
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      console.error('Translation processing failed:', error);
      return {
        success: false,
        error: {
          code: error instanceof ModelManagerError ? 'MODEL_MANAGER_ERROR' : 'PROCESSING_ERROR',
          message: error.message,
          type: error.constructor.name,
          details: error instanceof ModelManagerError ? error.details : {}
        },
        timestamp: new Date().toISOString()
      };
    }
  }

  private async handleTextToText(request: TranslationRequest, pipeline: PipelineConfig) {
    const modelName = this.getModelName(pipeline, 'nmt');
    
    const response = await this.modelManager.translateText({
      model: modelName,
      sourceLang: request.sourceLang,
      targetLang: request.targetLang,
      text: request.input.text || '',
      options: request.options
    });

    return {
      translatedText: response.translation.text,
      confidence: response.translation.confidence,
      alternatives: response.translation.alternatives,
      metadata: response.metadata
    };
  }

  private async handleTextToSpeech(request: TranslationRequest, pipeline: PipelineConfig) {
    // First translate text if needed
    let textToSynthesize = request.input.text || '';
    
    if (request.sourceLang !== request.targetLang) {
      const nmtModel = this.getModelName(pipeline, 'nmt');
      const translationResponse = await this.modelManager.translateText({
        model: nmtModel,
        sourceLang: request.sourceLang,
        targetLang: request.targetLang,
        text: textToSynthesize,
        options: request.options
      });
      textToSynthesize = translationResponse.translation.text;
    }

    // Then synthesize speech
    const ttsModel = this.getModelName(pipeline, 'tts');
    const synthResponse = await this.modelManager.synthesizeSpeech({
      model: ttsModel,
      text: textToSynthesize,
      language: request.targetLang,
      options: request.options
    });

    return {
      translatedText: textToSynthesize,
      audioData: synthResponse.synthesis.audioBase64,
      audioFormat: synthResponse.synthesis.format,
      duration: synthResponse.synthesis.duration,
      metadata: synthResponse.metadata
    };
  }

  private async handleSpeechToText(request: TranslationRequest, pipeline: PipelineConfig) {
    const asrModel = this.getModelName(pipeline, 'asr');
    
    // Transcribe audio
    const transcribeResponse = await this.modelManager.transcribeAudio(
      request.input.audioData, 
      {
        model: asrModel,
        language: request.sourceLang,
        options: request.options
      }
    );

    let finalText = transcribeResponse.transcription.text;

    // Translate if needed
    if (request.sourceLang !== request.targetLang) {
      const nmtModel = this.getModelName(pipeline, 'nmt');
      const translationResponse = await this.modelManager.translateText({
        model: nmtModel,
        sourceLang: request.sourceLang,
        targetLang: request.targetLang,
        text: finalText,
        options: request.options
      });
      finalText = translationResponse.translation.text;
    }

    return {
      transcribedText: transcribeResponse.transcription.text,
      translatedText: finalText,
      confidence: transcribeResponse.transcription.confidence,
      segments: transcribeResponse.transcription.segments,
      metadata: transcribeResponse.metadata
    };
  }

  private async handleSpeechToSpeech(request: TranslationRequest, pipeline: PipelineConfig) {
    const asrModel = this.getModelName(pipeline, 'asr');
    const nmtModel = this.getModelName(pipeline, 'nmt');
    const ttsModel = this.getModelName(pipeline, 'tts');

    // 1. Speech to Text
    const transcribeResponse = await this.modelManager.transcribeAudio(
      request.input.audioData,
      {
        model: asrModel,
        language: request.sourceLang,
        options: request.options
      }
    );

    // 2. Text to Text (Translation)
    const translationResponse = await this.modelManager.translateText({
      model: nmtModel,
      sourceLang: request.sourceLang,
      targetLang: request.targetLang,
      text: transcribeResponse.transcription.text,
      options: request.options
    });

    // 3. Text to Speech
    const synthResponse = await this.modelManager.synthesizeSpeech({
      model: ttsModel,
      text: translationResponse.translation.text,
      language: request.targetLang,
      options: request.options
    });

    return {
      transcribedText: transcribeResponse.transcription.text,
      translatedText: translationResponse.translation.text,
      audioData: synthResponse.synthesis.audioBase64,
      audioFormat: synthResponse.synthesis.format,
      duration: synthResponse.synthesis.duration,
      confidence: transcribeResponse.transcription.confidence,
      metadata: {
        asr: transcribeResponse.metadata,
        nmt: translationResponse.metadata,
        tts: synthResponse.metadata
      }
    };
  }

  private getModelName(pipeline: PipelineConfig, modelType: 'asr' | 'nmt' | 'tts'): string {
    return pipeline.models[modelType]?.id || `default-${modelType}`;
  }

  private generateEvaluation(request: TranslationRequest, result: any) {
    // Mock evaluation metrics - in real implementation, this would calculate actual metrics
    return {
      bleuScore: Math.random() * 0.3 + 0.7, // 0.7-1.0
      cometScore: Math.random() * 0.3 + 0.7,
      wer: request.type.includes('speech') ? Math.random() * 0.1 : undefined,
      mosScore: request.type.includes('speech') ? Math.random() * 1 + 4 : undefined,
      metrics: {
        precision: Math.random() * 0.2 + 0.8,
        recall: Math.random() * 0.2 + 0.8,
        f1Score: Math.random() * 0.2 + 0.8
      }
    };
  }

  private convertPipelineFormat(modelManagerPipeline: any): PipelineConfig {
    return {
      id: modelManagerPipeline.id,
      name: modelManagerPipeline.name,
      description: modelManagerPipeline.description,
      models: {
        asr: {
          id: modelManagerPipeline.models?.asr?.name || 'whisper-base',
          name: modelManagerPipeline.models?.asr?.name || 'Whisper Base',
          version: modelManagerPipeline.models?.asr?.version || '1.0',
          type: 'asr'
        },
        nmt: {
          id: modelManagerPipeline.models?.nmt?.name || 'opus-mt',
          name: modelManagerPipeline.models?.nmt?.name || 'OPUS-MT',
          version: modelManagerPipeline.models?.nmt?.version || '1.0',
          type: 'nmt'
        },
        tts: {
          id: modelManagerPipeline.models?.tts?.name || 'espeak-ng',
          name: modelManagerPipeline.models?.tts?.name || 'eSpeak NG',
          version: modelManagerPipeline.models?.tts?.version || '1.0',
          type: 'tts'
        }
      }
    };
  }

  private getMockPipelines(): PipelineConfig[] {
    return [
      {
        id: 'baseline',
        name: 'Baseline Pipeline',
        description: 'Basic ASR → NMT → TTS pipeline',
        models: {
          asr: { id: 'whisper-base', name: 'Whisper Base', version: '1.0', type: 'asr' },
          nmt: { id: 'opus-mt', name: 'OPUS-MT', version: '1.0', type: 'nmt' },
          tts: { id: 'espeak-ng', name: 'eSpeak NG', version: '1.0', type: 'tts' }
        }
      },
      {
        id: 'advanced',
        name: 'Advanced Pipeline',
        description: 'Fine-tuned models for better quality',
        models: {
          asr: { id: 'whisper-large', name: 'Whisper Large', version: '2.0', type: 'asr' },
          nmt: { id: 'mbart-large', name: 'mBART Large', version: '1.5', type: 'nmt' },
          tts: { id: 'tacotron2', name: 'Tacotron 2', version: '1.2', type: 'tts' }
        }
      },
      {
        id: 'experimental',
        name: 'Experimental Pipeline',
        description: 'Latest experimental models',
        models: {
          asr: { id: 'wav2vec2-xl', name: 'Wav2Vec2 XL', version: '3.0', type: 'asr' },
          nmt: { id: 'nllb-200', name: 'NLLB-200', version: '2.0', type: 'nmt' },
          tts: { id: 'bark', name: 'Bark', version: '1.0', type: 'tts' }
        }
      }
    ];
  }
}

export default TranslationService;