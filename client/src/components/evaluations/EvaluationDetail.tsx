'use client';

import { Calendar, Download, Info, ExternalLink, Languages as LangIcon } from 'lucide-react';
import { EvaluationDetail as EvaluationDetailType } from '@/lib/evaluation-client';
import evaluationClient from '@/lib/evaluation-client';

interface EvaluationDetailProps {
  evaluation: EvaluationDetailType;
}

export default function EvaluationDetail({ evaluation }: EvaluationDetailProps) {
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
              <div className="font-medium text-gray-900 dark:text-white text-sm">
                {evaluation.nmt_model}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Info className="w-5 h-5 text-gray-500" />
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400">TTS Model</div>
              <div className="font-medium text-gray-900 dark:text-white text-sm">
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
                {metric.toUpperCase()}
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

      {/* Languages Section */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <LangIcon className="w-6 h-6" />
          Language Results
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(evaluation.languages).map(([language, info]) => (
            <div
              key={language}
              className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 border border-gray-200 dark:border-gray-700"
            >
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3 capitalize">
                {language}
              </h3>

              <div className="space-y-3 mb-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 dark:text-gray-400">Total Samples</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {info.total_samples}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 dark:text-gray-400">Valid Samples</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {info.valid_samples}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600 dark:text-gray-400">Success Rate</span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {((info.valid_samples / info.total_samples) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

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

      {/* Overall Summary */}
      {evaluation.overall_summary && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Overall Summary
          </h2>
          <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
            <pre className="text-sm text-gray-700 dark:text-gray-300 overflow-x-auto">
              {JSON.stringify(evaluation.overall_summary, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
