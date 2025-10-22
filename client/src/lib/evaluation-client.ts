/**
 * Evaluation API Client
 * Handles communication with the evaluation endpoints
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

// Types
export interface EvaluationSummary {
  run_id: string;
  timestamp: string;
  translation_type: string | null;
  language_pair: string;
  total_samples: number;
  valid_samples: number;
  metrics_computed: string[];
  aggregate_scores: Record<string, number>;
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
  run_id: string;
  timestamp: string;
  translation_type: string | null;
  language_pair: string;
  total_samples: number;
  valid_samples: number;
  skipped_samples: number;
  metrics_computed: string[];
  aggregate_scores: Record<string, number>;
  score_statistics: Record<string, ScoreStatistics>;
  per_sample_results: SampleResult[];
  visualizations: string[];
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
   * Get detailed results for a specific evaluation run
   */
  async getEvaluationById(runId: string): Promise<EvaluationDetail> {
    try {
      const response = await this.client.get<EvaluationDetail>(`/evaluations/${runId}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get URL for a visualization file
   */
  getVisualizationUrl(runId: string, filename: string): string {
    return `${this.client.defaults.baseURL}/evaluations/${runId}/files/${filename}`;
  }

  /**
   * Get URL for the HTML report
   */
  getReportUrl(runId: string): string {
    return `${this.client.defaults.baseURL}/evaluations/${runId}/files/summary_report.html`;
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
