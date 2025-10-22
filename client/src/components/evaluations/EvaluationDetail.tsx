'use client';

import { Calendar, Languages, Download, Info, ExternalLink } from 'lucide-react';
import { EvaluationDetail as EvaluationDetailType } from '@/lib/evaluation-client';
import MetricCard from './MetricCard';
import evaluationClient from '@/lib/evaluation-client';

interface EvaluationDetailProps {
  evaluation: EvaluationDetailType;
}

// Metric metadata for proper display
const METRIC_INFO: Record<string, { description: string; higherIsBetter: boolean; goodRange?: [number, number] }> = {
  bleu: {
    description: 'N-gram overlap with reference',
    higherIsBetter: true,
    goodRange: [20, 40],
  },
  chrf: {
    description: 'Character n-gram F-score',
    higherIsBetter: true,
    goodRange: [40, 60],
  },
  comet: {
    description: 'Neural semantic quality',
    higherIsBetter: true,
    goodRange: [0.6, 0.8],
  },
  mcd: {
    description: 'Mel-cepstral distortion (dB)',
    higherIsBetter: false,
    goodRange: [4, 6],
  },
  blaser: {
    description: 'Speech translation quality',
    higherIsBetter: true,
    goodRange: [3.5, 4.5],
  },
};

export default function EvaluationDetail({ evaluation }: EvaluationDetailProps) {
  const handleOpenReport = () => {
    window.open(evaluationClient.getReportUrl(evaluation.run_id), '_blank');
  };

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          {evaluation.run_id}
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
            <Languages className="w-5 h-5 text-gray-500" />
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Translation</div>
              <div className="font-medium text-gray-900 dark:text-white">
                {evaluation.translation_type || 'Unknown'} • {evaluation.language_pair}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Info className="w-5 h-5 text-gray-500" />
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Samples</div>
              <div className="font-medium text-gray-900 dark:text-white">
                {evaluation.valid_samples} valid / {evaluation.total_samples} total
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 flex gap-3">
          <button
            onClick={handleOpenReport}
            className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
            Open Full Report
          </button>
        </div>
      </div>

      {/* Metrics Cards */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Aggregate Metrics
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Object.entries(evaluation.aggregate_scores).map(([metric, score]) => {
            const info = METRIC_INFO[metric] || {
              description: metric.toUpperCase(),
              higherIsBetter: true,
            };

            return (
              <MetricCard
                key={metric}
                name={metric.toUpperCase()}
                value={score}
                description={info.description}
                higherIsBetter={info.higherIsBetter}
                goodRange={info.goodRange}
              />
            );
          })}
        </div>
      </div>

      {/* Visualizations */}
      {evaluation.visualizations && evaluation.visualizations.length > 0 && (
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Visualizations
          </h2>

          <div className="space-y-6">
            {/* Quality Dashboard (highest priority) */}
            {evaluation.visualizations.includes('quality_dashboard.png') && (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Quality Dashboard
                </h3>
                <img
                  src={evaluationClient.getVisualizationUrl(evaluation.run_id, 'quality_dashboard.png')}
                  alt="Quality Dashboard"
                  className="w-full rounded-lg border border-gray-200 dark:border-gray-700"
                />
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-3">
                  Comprehensive quality analysis showing distribution, trends, and categorization.
                </p>
              </div>
            )}

            {/* Normalized Metrics */}
            {evaluation.visualizations.includes('normalized_metrics.png') && (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Normalized Metrics Comparison
                </h3>
                <img
                  src={evaluationClient.getVisualizationUrl(evaluation.run_id, 'normalized_metrics.png')}
                  alt="Normalized Metrics"
                  className="w-full rounded-lg border border-gray-200 dark:border-gray-700"
                />
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-3">
                  All metrics normalized to 0-1 scale with expected quality ranges indicated.
                </p>
              </div>
            )}

            {/* Other visualizations */}
            {evaluation.visualizations
              .filter((viz) => !viz.includes('quality_dashboard') && !viz.includes('normalized_metrics'))
              .slice(0, 4) // Limit to avoid too many images
              .map((viz) => (
                <div key={viz} className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    {viz.replace('.png', '').replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                  </h3>
                  <img
                    src={evaluationClient.getVisualizationUrl(evaluation.run_id, viz)}
                    alt={viz}
                    className="w-full rounded-lg border border-gray-200 dark:border-gray-700"
                  />
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Statistics Table */}
      {evaluation.score_statistics && Object.keys(evaluation.score_statistics).length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Score Statistics
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">Metric</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-900 dark:text-white">Mean</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-900 dark:text-white">Std Dev</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-900 dark:text-white">Min</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-900 dark:text-white">Max</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(evaluation.score_statistics).map(([metric, stats]) => (
                  <tr key={metric} className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="py-3 px-4 font-medium text-gray-900 dark:text-white">{metric.toUpperCase()}</td>
                    <td className="py-3 px-4 text-right text-gray-700 dark:text-gray-300">{stats.mean.toFixed(3)}</td>
                    <td className="py-3 px-4 text-right text-gray-700 dark:text-gray-300">{stats.std.toFixed(3)}</td>
                    <td className="py-3 px-4 text-right text-gray-700 dark:text-gray-300">{stats.min.toFixed(3)}</td>
                    <td className="py-3 px-4 text-right text-gray-700 dark:text-gray-300">{stats.max.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
