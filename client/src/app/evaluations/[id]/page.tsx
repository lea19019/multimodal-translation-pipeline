'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { ArrowLeft, Loader2, AlertCircle } from 'lucide-react';
import evaluationClient, { EvaluationDetail as EvaluationDetailType } from '@/lib/evaluation-client';
import EvaluationDetailComponent from '@/components/evaluations/EvaluationDetail';

export default function EvaluationDetailPage() {
  const params = useParams();
  const router = useRouter();
  const runId = params.id as string;

  const [evaluation, setEvaluation] = useState<EvaluationDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadEvaluation();
  }, [runId]);

  const loadEvaluation = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await evaluationClient.getEvaluationById(runId);
      setEvaluation(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load evaluation');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-center min-h-[400px]">
            <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
          </div>
        </div>
      </main>
    );
  }

  if (error || !evaluation) {
    return (
      <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
        <div className="container mx-auto px-4 py-8">
          <div className="max-w-2xl mx-auto">
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
              <div className="flex items-center gap-3">
                <AlertCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
                <div>
                  <h3 className="font-semibold text-red-900 dark:text-red-100">Error Loading Evaluation</h3>
                  <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                    {error || 'Evaluation not found'}
                  </p>
                </div>
              </div>
              <div className="mt-4 flex gap-3">
                <button
                  onClick={loadEvaluation}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                  Retry
                </button>
                <button
                  onClick={() => router.push('/evaluations')}
                  className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                >
                  Back to List
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Back Button */}
        <button
          onClick={() => router.push('/evaluations')}
          className="flex items-center gap-2 text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 mb-6 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span className="font-medium">Back to Evaluations</span>
        </button>

        {/* Evaluation Detail */}
        <div className="max-w-7xl mx-auto">
          <EvaluationDetailComponent evaluation={evaluation} />
        </div>
      </div>
    </main>
  );
}
