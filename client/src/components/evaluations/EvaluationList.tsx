'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Calendar, Languages, BarChart3, Loader2, AlertCircle } from 'lucide-react';
import evaluationClient, { EvaluationSummary } from '@/lib/evaluation-client';

export default function EvaluationList() {
  const [evaluations, setEvaluations] = useState<EvaluationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadEvaluations();
  }, []);

  const loadEvaluations = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await evaluationClient.getEvaluations();
      setEvaluations(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load evaluations');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
        <div className="flex items-center gap-3">
          <AlertCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
          <div>
            <h3 className="font-semibold text-red-900 dark:text-red-100">Error Loading Evaluations</h3>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">{error}</p>
          </div>
        </div>
        <button
          onClick={loadEvaluations}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (evaluations.length === 0) {
    return (
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-12 text-center">
        <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          No Evaluations Found
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Run an evaluation to see results here
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {evaluations.map((evaluation) => (
        <Link
          key={evaluation.execution_id}
          href={`/evaluations/${evaluation.execution_id}`}
          className="block"
        >
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-shadow border border-gray-200 dark:border-gray-700 p-6 h-full">
            {/* Header */}
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                {evaluation.execution_id}
              </h3>
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                <Calendar className="w-4 h-4" />
                <span>{new Date(evaluation.timestamp).toLocaleString()}</span>
              </div>
            </div>

            {/* Model Info */}
            <div className="mb-4 space-y-1 text-sm">
              <div className="text-gray-600 dark:text-gray-400">
                <span className="font-medium">NMT:</span> {evaluation.nmt_model}
              </div>
              <div className="text-gray-600 dark:text-gray-400">
                <span className="font-medium">TTS:</span> {evaluation.tts_model}
              </div>
            </div>

            {/* Languages */}
            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4">
              <Languages className="w-4 h-4" />
              <span>
                {Object.keys(evaluation.languages).length} language(s) • {evaluation.metrics.map(m => m.toUpperCase()).join(', ')}
              </span>
            </div>

            {/* Samples Count */}
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3 mb-4">
              <div className="text-sm text-gray-600 dark:text-gray-400">Total Samples</div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {evaluation.total_valid_samples} / {evaluation.total_samples}
              </div>
            </div>

            {/* Overall Scores Preview */}
            {evaluation.overall_summary?.average_scores && (
              <div className="grid grid-cols-3 gap-2 mb-4">
                {Object.entries(evaluation.overall_summary.average_scores).slice(0, 3).map(([metric, stats]) => (
                  <div key={metric} className="bg-primary-50 dark:bg-primary-900/20 rounded-lg p-2 text-center">
                    <div className="text-xs text-primary-600 dark:text-primary-400 font-medium">{metric.toUpperCase()}</div>
                    <div className="text-sm font-bold text-primary-700 dark:text-primary-300">
                      {typeof stats === 'object' && stats.mean ? stats.mean.toFixed(2) : '-'}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Languages Summary */}
            <div className="space-y-2">
              {Object.entries(evaluation.languages).slice(0, 3).map(([lang, info]) => (
                <div key={lang} className="flex justify-between items-center text-sm">
                  <span className="text-gray-600 dark:text-gray-400 capitalize">{lang}</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {info.valid_samples} / {info.total_samples}
                  </span>
                </div>
              ))}
              {Object.keys(evaluation.languages).length > 3 && (
                <div className="text-xs text-gray-500 dark:text-gray-400 text-center pt-2">
                  +{Object.keys(evaluation.languages).length - 3} more
                </div>
              )}
            </div>

            {/* View Button */}
            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
              <span className="text-primary-600 dark:text-primary-400 font-medium text-sm hover:underline">
                View Details →
              </span>
            </div>
          </div>
        </Link>
      ))}
    </div>
  );
}
