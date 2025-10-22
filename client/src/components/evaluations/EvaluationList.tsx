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
          key={evaluation.run_id}
          href={`/evaluations/${evaluation.run_id}`}
          className="block"
        >
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-lg transition-shadow border border-gray-200 dark:border-gray-700 p-6 h-full">
            {/* Header */}
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                {evaluation.run_id}
              </h3>
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                <Calendar className="w-4 h-4" />
                <span>{new Date(evaluation.timestamp).toLocaleString()}</span>
              </div>
            </div>

            {/* Translation Info */}
            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4">
              <Languages className="w-4 h-4" />
              <span>
                {evaluation.translation_type || 'Unknown'} • {evaluation.language_pair}
              </span>
            </div>

            {/* Samples Count */}
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3 mb-4">
              <div className="text-sm text-gray-600 dark:text-gray-400">Samples</div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {evaluation.valid_samples} / {evaluation.total_samples}
              </div>
            </div>

            {/* Metrics Summary */}
            <div className="space-y-2">
              {Object.entries(evaluation.aggregate_scores).map(([metric, score]) => (
                <div key={metric} className="flex justify-between items-center text-sm">
                  <span className="text-gray-600 dark:text-gray-400 uppercase">{metric}</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {score != null ? score.toFixed(2) : 'N/A'}
                  </span>
                </div>
              ))}
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
