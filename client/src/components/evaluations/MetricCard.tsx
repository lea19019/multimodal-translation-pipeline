'use client';

import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MetricCardProps {
  name: string;
  value: number | null;
  description?: string;
  higherIsBetter?: boolean;
  goodRange?: [number, number];
}

/**
 * Display a single metric with quality indication
 */
export default function MetricCard({
  name,
  value,
  description,
  higherIsBetter = true,
  goodRange,
}: MetricCardProps) {
  if (value === null || value === undefined) {
    return (
      <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6 border border-gray-200 dark:border-gray-600">
        <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">{name}</h3>
        <p className="text-2xl font-bold text-gray-400 dark:text-gray-500">N/A</p>
        {description && (
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">{description}</p>
        )}
      </div>
    );
  }

  // Determine quality based on good range
  let quality: 'excellent' | 'good' | 'fair' | 'poor' = 'fair';
  let qualityColor = 'text-yellow-600 dark:text-yellow-400';
  let bgColor = 'bg-yellow-50 dark:bg-yellow-900/20';
  let borderColor = 'border-yellow-200 dark:border-yellow-800';

  if (goodRange) {
    const [goodMin, goodMax] = goodRange;
    if (value >= goodMin && value <= goodMax) {
      quality = 'good';
      qualityColor = 'text-green-600 dark:text-green-400';
      bgColor = 'bg-green-50 dark:bg-green-900/20';
      borderColor = 'border-green-200 dark:border-green-800';
    } else if (
      (higherIsBetter && value > goodMax) ||
      (!higherIsBetter && value < goodMin)
    ) {
      quality = 'excellent';
      qualityColor = 'text-emerald-600 dark:text-emerald-400';
      bgColor = 'bg-emerald-50 dark:bg-emerald-900/20';
      borderColor = 'border-emerald-200 dark:border-emerald-800';
    } else {
      quality = 'poor';
      qualityColor = 'text-red-600 dark:text-red-400';
      bgColor = 'bg-red-50 dark:bg-red-900/20';
      borderColor = 'border-red-200 dark:border-red-800';
    }
  }

  const Icon = higherIsBetter ? TrendingUp : TrendingDown;

  return (
    <div className={`${bgColor} rounded-lg p-6 border ${borderColor}`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-600 dark:text-gray-300">{name}</h3>
        <Icon className={`w-5 h-5 ${qualityColor}`} />
      </div>

      <p className={`text-3xl font-bold ${qualityColor}`}>
        {value.toFixed(2)}
      </p>

      {description && (
        <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">{description}</p>
      )}

      <div className="mt-3">
        <span
          className={`inline-block px-2 py-1 text-xs font-semibold rounded ${qualityColor}`}
        >
          {quality.charAt(0).toUpperCase() + quality.slice(1)}
        </span>
      </div>
    </div>
  );
}
