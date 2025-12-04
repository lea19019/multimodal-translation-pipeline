/**
 * Evaluation API Client
 * Handles communication with the evaluation endpoints
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

// Types
export interface LanguageInfo {
  total_samples: number;
  valid_samples: number;
  aggregate_scores?: Record<string, number>;
}

export interface OverallSummary {
  languages_evaluated: number;
  average_scores: Record<string, {
    mean: number;
    median: number;
    std: number;
    min: number;
    max: number;
  }>;
}

export interface EvaluationSummary {
  execution_id: string;
  timestamp: string;
  nmt_model: string;
  tts_model: string;
  metrics: string[];
  languages: Record<string, LanguageInfo>;
  total_samples: number;
  total_valid_samples: number;
  overall_summary?: OverallSummary;
}

export interface LanguageResults {
  language: string;
  summary: Record<string, any>;
  visualizations: string[];
}

export interface SampleResult {
  uuid: string;
  source_language: string | null;
  target_language: string | null;
  [key: string]: any; // For metric scores like bleu_score, comet_score, etc.
}

export interface ScoreStatistics {
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface EvaluationDetail {
  execution_id: string;
  timestamp: string;
  nmt_model: string;
  tts_model: string;
  metrics: string[];
  languages: Record<string, LanguageInfo>;
  total_samples: number;
  total_valid_samples: number;
  overall_summary?: OverallSummary;
}

interface APIError {
  detail: string;
}

/**
 * Client for interacting with evaluation API endpoints
 */
export class EvaluationClient {
  private client: AxiosInstance;

  constructor(baseUrl: string = process.env.NEXT_PUBLIC_GATEWAY_URL || 'http://localhost:8075') {
    this.client = axios.create({
      baseURL: baseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * List all evaluation runs
   */
  async getEvaluations(): Promise<EvaluationSummary[]> {
    try {
      const response = await this.client.get<EvaluationSummary[]>('/evaluations');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get detailed results for a specific evaluation execution
   */
  async getEvaluationById(executionId: string): Promise<EvaluationDetail> {
    try {
      const response = await this.client.get<EvaluationDetail>(`/evaluations/${executionId}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get results for a specific language within an execution
   */
  async getLanguageResults(executionId: string, language: string): Promise<LanguageResults> {
    try {
      const response = await this.client.get<LanguageResults>(
        `/evaluations/${executionId}/languages/${language}`
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get URL for a language-specific visualization file
   */
  getLanguageVisualizationUrl(executionId: string, language: string, filename: string): string {
    return `${this.client.defaults.baseURL}/evaluations/${executionId}/languages/${language}/files/${filename}`;
  }

  /**
   * Get list of execution-level visualizations
   */
  async getExecutionVisualizations(executionId: string): Promise<{ execution_id: string; visualizations: string[] }> {
    try {
      const response = await this.client.get(`/evaluations/${executionId}/visualizations`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get URL for an execution-level file (manifest, overall_summary, visualizations)
   */
  getExecutionFileUrl(executionId: string, filename: string): string {
    return `${this.client.defaults.baseURL}/evaluations/${executionId}/files/${filename}`;
  }

  /**
   * Get URL for a visualization file (deprecated, use getLanguageVisualizationUrl)
   */
  getVisualizationUrl(runId: string, filename: string): string {
    // For backward compatibility - assumes first language or general file
    return `${this.client.defaults.baseURL}/evaluations/${runId}/files/${filename}`;
  }

  /**
   * Get URL for the HTML report (deprecated, use getLanguageVisualizationUrl)
   */
  getReportUrl(runId: string): string {
    return `${this.client.defaults.baseURL}/evaluations/${runId}/files/summary_report.html`;
  }

  /**
   * Get URL for a language-specific HTML report
   */
  getLanguageReportUrl(executionId: string, language: string): string {
    return this.getLanguageVisualizationUrl(executionId, language, 'summary_report.html');
  }

  /**
   * Handle API errors consistently
   */
  private handleError(error: unknown): Error {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<APIError>;

      if (axiosError.response?.data?.detail) {
        return new Error(axiosError.response.data.detail);
      }

      if (axiosError.response?.status === 404) {
        return new Error('Evaluation not found');
      }

      if (axiosError.response?.status === 500) {
        return new Error('Server error occurred');
      }

      if (axiosError.code === 'ECONNREFUSED') {
        return new Error('Cannot connect to evaluation service');
      }

      if (axiosError.code === 'ETIMEDOUT') {
        return new Error('Request timed out');
      }
    }

    return new Error('An unknown error occurred');
  }
}

// Export a default instance
const evaluationClient = new EvaluationClient();
export default evaluationClient;
