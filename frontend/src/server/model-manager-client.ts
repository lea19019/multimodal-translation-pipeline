import fetch from 'node-fetch';

interface ModelManagerConfig {
  baseUrl: string;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
}

class ModelManagerClient {
  private config: ModelManagerConfig;

  constructor(config: Partial<ModelManagerConfig> = {}) {
    this.config = {
      baseUrl: process.env.MODEL_MANAGER_URL || 'http://localhost:8000',
      timeout: parseInt(process.env.MODEL_MANAGER_TIMEOUT || '30000'),
      retryAttempts: parseInt(process.env.MODEL_MANAGER_RETRY_ATTEMPTS || '3'),
      retryDelay: parseInt(process.env.MODEL_MANAGER_RETRY_DELAY || '1000'),
      ...config
    };
  }

  async healthCheck(): Promise<any> {
    return this.makeRequest('GET', '/health');
  }

  async getModelStatus(): Promise<any> {
    return this.makeRequest('GET', '/models/status');
  }

  async loadModels(models: any[]): Promise<any> {
    return this.makeRequest('POST', '/models/load', { models });
  }

  async translateText(request: {
    model: string;
    sourceLang: string;
    targetLang: string;
    text: string;
    options?: any;
  }): Promise<any> {
    return this.makeRequest('POST', '/translate/text', request);
  }

  async transcribeAudio(audioBuffer: Buffer, options: {
    model: string;
    language?: string;
    task?: string;
    options?: any;
  }): Promise<any> {
    const FormData = require('form-data');
    const form = new FormData();
    
    form.append('model', options.model);
    form.append('language', options.language || 'auto');
    form.append('task', options.task || 'transcribe');
    form.append('audio', audioBuffer, { filename: 'audio.wav', contentType: 'audio/wav' });
    
    if (options.options) {
      form.append('options', JSON.stringify(options.options));
    }

    return this.makeRequestWithForm('POST', '/transcribe', form);
  }

  async synthesizeSpeech(request: {
    model: string;
    text: string;
    language?: string;
    voice?: string;
    options?: any;
  }): Promise<any> {
    return this.makeRequest('POST', '/synthesize', request);
  }

  async processBatch(request: {
    batchId: string;
    requests: any[];
    options?: any;
  }): Promise<any> {
    return this.makeRequest('POST', '/batch/process', request);
  }

  async getPipelines(): Promise<any> {
    return this.makeRequest('GET', '/pipelines');
  }

  async getPipeline(pipelineId: string): Promise<any> {
    return this.makeRequest('GET', `/pipelines/${pipelineId}`);
  }

  private async makeRequest(method: string, endpoint: string, data?: any): Promise<any> {
    const url = `${this.config.baseUrl}${endpoint}`;
    
    for (let attempt = 1; attempt <= this.config.retryAttempts; attempt++) {
      try {
        const response = await fetch(url, {
          method,
          headers: {
            'Content-Type': 'application/json',
          },
          body: data ? JSON.stringify(data) : undefined,
          timeout: this.config.timeout,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new ModelManagerError(
            `HTTP ${response.status}: ${response.statusText}`,
            response.status,
            errorData
          );
        }

        return await response.json();
      } catch (error) {
        if (attempt === this.config.retryAttempts) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          throw new ModelManagerError(
            `Failed after ${this.config.retryAttempts} attempts: ${errorMessage}`,
            0,
            { originalError: errorMessage }
          );
        }
        
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.warn(`Model Manager request attempt ${attempt} failed, retrying...`, errorMessage);
        await new Promise(resolve => setTimeout(resolve, this.config.retryDelay));
      }
    }
  }

  private async makeRequestWithForm(method: string, endpoint: string, form: any): Promise<any> {
    const url = `${this.config.baseUrl}${endpoint}`;
    
    for (let attempt = 1; attempt <= this.config.retryAttempts; attempt++) {
      try {
        const response = await fetch(url, {
          method,
          body: form,
          timeout: this.config.timeout,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new ModelManagerError(
            `HTTP ${response.status}: ${response.statusText}`,
            response.status,
            errorData
          );
        }

        return await response.json();
      } catch (error) {
        if (attempt === this.config.retryAttempts) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          throw new ModelManagerError(
            `Failed after ${this.config.retryAttempts} attempts: ${errorMessage}`,
            0,
            { originalError: errorMessage }
          );
        }
        
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.warn(`Model Manager request attempt ${attempt} failed, retrying...`, errorMessage);
        await new Promise(resolve => setTimeout(resolve, this.config.retryDelay));
      }
    }
  }
}

class ModelManagerError extends Error {
  constructor(
    message: string,
    public statusCode: number = 0,
    public details: any = {}
  ) {
    super(message);
    this.name = 'ModelManagerError';
  }
}

export { ModelManagerClient, ModelManagerError };
export default ModelManagerClient;