'use client';

import { useState, useEffect } from 'react';
import { Calendar, Download, Info, ExternalLink, Languages as LangIcon, BarChart3, TrendingUp, TrendingDown } from 'lucide-react';
import { EvaluationDetail as EvaluationDetailType } from '@/lib/evaluation-client';
import evaluationClient from '@/lib/evaluation-client';

interface EvaluationDetailProps {
  evaluation: EvaluationDetailType;
}

// Metric metadata for display
const METRIC_INFO: Record<string, { name: string; higherIsBetter: boolean; format: (v: number) => string }> = {
  bleu: { name: 'BLEU', higherIsBetter: true, format: (v) => v.toFixed(2) },
  chrf: { name: 'chrF++', higherIsBetter: true, format: (v) => v.toFixed(2) },
  comet: { name: 'COMET', higherIsBetter: true, format: (v) => v.toFixed(4) },
  mcd: { name: 'MCD', higherIsBetter: false, format: (v) => v.toFixed(2) },
  blaser: { name: 'BLASER', higherIsBetter: true, format: (v) => v.toFixed(3) },
};

export default function EvaluationDetail({ evaluation }: EvaluationDetailProps) {
  const [executionVizs, setExecutionVizs] = useState<string[]>([]);

  useEffect(() => {
    // Fetch execution-level visualizations
    evaluationClient.getExecutionVisualizations(evaluation.execution_id)
      .then((data) => setExecutionVizs(data.visualizations))
      .catch((err) => console.error('Failed to load visualizations:', err));
  }, [evaluation.execution_id]);

  const handleDownloadManifest = () => {
    window.open(evaluationClient.getExecutionFileUrl(evaluation.execution_id, 'manifest.json'), '_blank');
  };

  const handleDownloadOverallSummary = () => {
    window.open(evaluationClient.getExecutionFileUrl(evaluation.execution_id, 'overall_summary.json'), '_blank');
  };

  const handleOpenLanguageReport = (language: string) => {
    window.open(evaluationClient.getLanguageReportUrl(evaluation.execution_id, language), '_blank');
  };

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          {evaluation.execution_id}
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="flex items-center gap-3">
            <Calendar className="w-5 h-5 text-gray-500" />
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Timestamp</div>
              <div className="font-medium text-gray-900 dark:text-white">
                {new Date(evaluation.timestamp).toLocaleString()}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Info className="w-5 h-5 text-gray-500" />
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400">NMT Model</div>
              <div className="font-medium text-gray-900 dark:text-white text-sm truncate max-w-[200px]" title={evaluation.nmt_model}>
                {evaluation.nmt_model}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Info className="w-5 h-5 text-gray-500" />
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400">TTS Model</div>
              <div className="font-medium text-gray-900 dark:text-white text-sm truncate max-w-[200px]" title={evaluation.tts_model}>
                {evaluation.tts_model}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Info className="w-5 h-5 text-gray-500" />
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Samples</div>
              <div className="font-medium text-gray-900 dark:text-white">
                {evaluation.total_valid_samples} / {evaluation.total_samples}
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">Metrics Evaluated</div>
          <div className="flex flex-wrap gap-2">
            {evaluation.metrics.map((metric) => (
              <span
                key={metric}
                className="px-3 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm font-medium"
              >
                {METRIC_INFO[metric]?.name || metric.toUpperCase()}
              </span>
            ))}
          </div>
        </div>

        <div className="mt-4 flex gap-3">
          <button
            onClick={handleDownloadManifest}
            className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm"
          >
            <Download className="w-4 h-4" />
            Download Manifest
          </button>
          {evaluation.overall_summary && (
            <button
              onClick={handleDownloadOverallSummary}
              className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors text-sm"
            >
              <Download className="w-4 h-4" />
              Overall Summary
            </button>
          )}
        </div>
      </div>

      {/* Overall Average Scores */}
      {evaluation.overall_summary && evaluation.overall_summary.average_scores && (
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-6 h-6" />
            Overall Average Scores
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {Object.entries(evaluation.overall_summary.average_scores).map(([metric, stats]) => {
              const info = METRIC_INFO[metric] || { name: metric.toUpperCase(), higherIsBetter: true, format: (v: number) => v.toFixed(3) };
              return (
                <div
                  key={metric}
                  className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-gray-600 dark:text-gray-400">
                      {info.name}
                    </span>
                    {info.higherIsBetter ? (
                      <TrendingUp className="w-4 h-4 text-green-500" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-blue-500" />
                    )}
                  </div>
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">
                    {info.format(stats.mean)}
                  </div>
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 space-y-1">
                    <div className="flex justify-between">
                      <span>Median:</span>
                      <span>{info.format(stats.median)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Std:</span>
                      <span>±{stats.std.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Range:</span>
                      <span>{info.format(stats.min)} - {info.format(stats.max)}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Execution-level Visualizations */}
      {executionVizs.length > 0 && (
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Cross-Language Analysis
          </h2>
          <div className="grid grid-cols-1 gap-6">
            {executionVizs.map((viz) => (
              <div key={viz} className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3 capitalize">
                  {viz.replace('.png', '').replace(/_/g, ' ')}
                </h3>
                <img
                  src={evaluationClient.getExecutionFileUrl(evaluation.execution_id, viz)}
                  alt={viz}
                  className="w-full rounded-lg"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Languages Section */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <LangIcon className="w-6 h-6" />
          Per-Language Results
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {Object.entries(evaluation.languages).map(([language, info]) => (
            <div
              key={language}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 border border-gray-200 dark:border-gray-700"
            >
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white capitalize">
                  {language}
                </h3>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {info.valid_samples} / {info.total_samples} samples
                </span>
              </div>

              {/* Aggregate scores for this language */}
              {info.aggregate_scores && (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">
                  {Object.entries(info.aggregate_scores).map(([metric, score]) => {
                    const metricInfo = METRIC_INFO[metric] || { name: metric.toUpperCase(), higherIsBetter: true, format: (v: number) => v.toFixed(3) };
                    return (
                      <div key={metric} className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                        <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                          {metricInfo.name}
                        </div>
                        <div className="text-lg font-bold text-gray-900 dark:text-white">
                          {metricInfo.format(score)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              <div className="flex gap-2">
                <button
                  onClick={() => handleOpenLanguageReport(language)}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm"
                >
                  <ExternalLink className="w-4 h-4" />
                  View Report
                </button>
                <a
                  href={evaluationClient.getLanguageVisualizationUrl(
                    evaluation.execution_id,
                    language,
                    'detailed_results.csv'
                  )}
                  download
                  className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors text-sm"
                >
                  <Download className="w-4 h-4" />
                  CSV
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
