'use client';

import { useState, useEffect } from 'react';
import { CheckCircle2, XCircle, AlertCircle, Loader2 } from 'lucide-react';
import { MultimodalTranslationClient, HealthResponse } from '@/lib/api-client';

export default function HealthStatus() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const client = new MultimodalTranslationClient(
        process.env.NEXT_PUBLIC_GATEWAY_URL,
        process.env.NEXT_PUBLIC_ASR_URL,
        process.env.NEXT_PUBLIC_NMT_URL,
        process.env.NEXT_PUBLIC_TTS_URL
      );
      const healthData = await client.healthCheck();
      setHealth(healthData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check health');
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !health) {
    return (
      <div className="max-w-4xl mx-auto bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-center gap-2 text-gray-600 dark:text-gray-300">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span>Checking services...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto bg-red-50 dark:bg-red-900/20 rounded-lg shadow-lg p-6">
        <div className="flex items-center gap-3">
          <XCircle className="w-6 h-6 text-red-600 dark:text-red-400 flex-shrink-0" />
          <div className="flex-1">
            <h3 className="font-semibold text-red-900 dark:text-red-100">
              Services Unavailable
            </h3>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">{error}</p>
          </div>
          <button
            onClick={checkHealth}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!health) return null;

  const allHealthy =
    health.status === 'healthy' &&
    health.downstream_services?.asr?.status === 'healthy' &&
    health.downstream_services?.nmt?.status === 'healthy' &&
    health.downstream_services?.tts?.status === 'healthy';

  return (
    <div className="max-w-4xl mx-auto bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
          {allHealthy ? (
            <>
              <CheckCircle2 className="w-6 h-6 text-green-600" />
              All Services Online
            </>
          ) : (
            <>
              <AlertCircle className="w-6 h-6 text-yellow-600" />
              Some Services Offline
            </>
          )}
        </h3>
        <button
          onClick={checkHealth}
          disabled={loading}
          className="px-3 py-1 text-sm bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors disabled:opacity-50"
        >
          {loading ? 'Checking...' : 'Refresh'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* ASR Service */}
        <div className="flex items-center gap-3 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          {health.downstream_services?.asr?.status === 'healthy' ? (
            <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0" />
          ) : (
            <XCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
          )}
          <div>
            <div className="font-medium text-gray-900 dark:text-white">ASR</div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              Speech Recognition
            </div>
          </div>
        </div>

        {/* NMT Service */}
        <div className="flex items-center gap-3 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          {health.downstream_services?.nmt?.status === 'healthy' ? (
            <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0" />
          ) : (
            <XCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
          )}
          <div>
            <div className="font-medium text-gray-900 dark:text-white">NMT</div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              Translation
            </div>
          </div>
        </div>

        {/* TTS Service */}
        <div className="flex items-center gap-3 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          {health.downstream_services?.tts?.status === 'healthy' ? (
            <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0" />
          ) : (
            <XCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
          )}
          <div>
            <div className="font-medium text-gray-900 dark:text-white">TTS</div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              Speech Synthesis
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
