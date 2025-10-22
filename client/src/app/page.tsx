'use client';

import { useState } from 'react';
import Link from 'next/link';
import {
  Languages,
  Mic,
  FileAudio,
  MessageSquare,
  CheckCircle2,
  XCircle,
  Loader2,
  BarChart3
} from 'lucide-react';
import TranslationForm from '@/components/TranslationForm';
import HealthStatus from '@/components/HealthStatus';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'text' | 'audio-to-text' | 'text-to-audio' | 'audio-to-audio'>('text');

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Navigation */}
        <nav className="flex justify-end mb-4">
          <Link
            href="/evaluations"
            className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg shadow-md hover:shadow-lg transition-all hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            <BarChart3 className="w-5 h-5" />
            <span className="font-medium">View Evaluations</span>
          </Link>
        </nav>

        {/* Header */}
        <header className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Languages className="w-12 h-12 text-primary-600" />
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
              Multimodal Translation
            </h1>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Translate text and speech across languages with AI
          </p>
        </header>

        {/* Health Status */}
        <div className="mb-8">
          <HealthStatus />
        </div>

        {/* Translation Mode Tabs */}
        <div className="max-w-4xl mx-auto mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-2">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
              <button
                onClick={() => setActiveTab('text')}
                className={`flex items-center justify-center gap-2 px-4 py-3 rounded-lg transition-all ${
                  activeTab === 'text'
                    ? 'bg-primary-600 text-white shadow-md'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                <MessageSquare className="w-5 h-5" />
                <span className="font-medium">Text to Text</span>
              </button>

              <button
                onClick={() => setActiveTab('audio-to-text')}
                className={`flex items-center justify-center gap-2 px-4 py-3 rounded-lg transition-all ${
                  activeTab === 'audio-to-text'
                    ? 'bg-primary-600 text-white shadow-md'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                <Mic className="w-5 h-5" />
                <span className="font-medium">Audio to Text</span>
              </button>

              <button
                onClick={() => setActiveTab('text-to-audio')}
                className={`flex items-center justify-center gap-2 px-4 py-3 rounded-lg transition-all ${
                  activeTab === 'text-to-audio'
                    ? 'bg-primary-600 text-white shadow-md'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                <FileAudio className="w-5 h-5" />
                <span className="font-medium">Text to Audio</span>
              </button>

              <button
                onClick={() => setActiveTab('audio-to-audio')}
                className={`flex items-center justify-center gap-2 px-4 py-3 rounded-lg transition-all ${
                  activeTab === 'audio-to-audio'
                    ? 'bg-primary-600 text-white shadow-md'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                <Languages className="w-5 h-5" />
                <span className="font-medium">Audio to Audio</span>
              </button>
            </div>
          </div>
        </div>

        {/* Translation Form */}
        <TranslationForm mode={activeTab} />

        {/* Footer */}
        <footer className="mt-16 text-center text-gray-600 dark:text-gray-400">
          <p className="mb-2">
            Powered by Whisper (ASR) • NLLB (NMT) • Coqui XTTS (TTS)
          </p>
          <p className="text-sm">
            Open source multimodal translation platform
          </p>
        </footer>
      </div>
    </main>
  );
}
