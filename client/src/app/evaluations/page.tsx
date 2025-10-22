'use client';

import Link from 'next/link';
import { BarChart3, Home } from 'lucide-react';
import EvaluationList from '@/components/evaluations/EvaluationList';

export default function EvaluationsPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Back to Home */}
        <div className="mb-6">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
          >
            <Home className="w-5 h-5" />
            <span className="font-medium">Back to Translation</span>
          </Link>
        </div>

        {/* Header */}
        <header className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <BarChart3 className="w-12 h-12 text-primary-600" />
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
              Translation Evaluations
            </h1>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Review quality metrics and performance analysis of translation runs
          </p>
        </header>

        {/* Evaluation List */}
        <div className="max-w-7xl mx-auto">
          <EvaluationList />
        </div>
      </div>
    </main>
  );
}
