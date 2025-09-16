import React, { useState, useEffect } from 'react';
import { ArrowRight, Languages, Volume2 } from 'lucide-react';

const LanguageSelector = ({
  sourceLanguage,
  targetLanguage,
  onSourceLanguageChange,
  onTargetLanguageChange,
  outputFormat,
  onOutputFormatChange
}) => {
  const [languages] = useState({
    'auto': 'Auto Detect',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi'
  });

  const swapLanguages = () => {
    if (sourceLanguage !== 'auto') {
      onSourceLanguageChange(targetLanguage);
      onTargetLanguageChange(sourceLanguage);
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Language Settings</h2>
      
      {/* Language Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
        {/* Source Language */}
        <div>
          <label htmlFor="source-language" className="block text-sm font-medium text-gray-700 mb-2">
            From
          </label>
          <select
            id="source-language"
            value={sourceLanguage}
            onChange={(e) => onSourceLanguageChange(e.target.value)}
            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          >
            {Object.entries(languages).map(([code, name]) => (
              <option key={code} value={code}>
                {name}
              </option>
            ))}
          </select>
        </div>

        {/* Swap Button */}
        <div className="flex justify-center">
          <button
            onClick={swapLanguages}
            disabled={sourceLanguage === 'auto'}
            className="p-2 rounded-full border-2 border-gray-300 hover:border-blue-500 hover:bg-blue-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Swap languages"
          >
            <ArrowRight size={20} className="text-gray-600" />
          </button>
        </div>

        {/* Target Language */}
        <div>
          <label htmlFor="target-language" className="block text-sm font-medium text-gray-700 mb-2">
            To
          </label>
          <select
            id="target-language"
            value={targetLanguage}
            onChange={(e) => onTargetLanguageChange(e.target.value)}
            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          >
            {Object.entries(languages)
              .filter(([code]) => code !== 'auto')
              .map(([code, name]) => (
                <option key={code} value={code}>
                  {name}
                </option>
              ))}
          </select>
        </div>
      </div>

      {/* Output Format Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Output Format
        </label>
        <div className="flex gap-4">
          <label className="flex items-center">
            <input
              type="radio"
              name="output-format"
              value="text"
              checked={outputFormat === 'text'}
              onChange={(e) => onOutputFormatChange(e.target.value)}
              className="mr-2"
            />
            <Languages size={16} className="mr-1" />
            Text Only
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              name="output-format"
              value="audio"
              checked={outputFormat === 'audio'}
              onChange={(e) => onOutputFormatChange(e.target.value)}
              className="mr-2"
            />
            <Volume2 size={16} className="mr-1" />
            Text + Audio
          </label>
        </div>
      </div>

      {/* Language Information */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start gap-2">
          <Languages className="h-5 w-5 text-blue-600 mt-0.5" />
          <div className="text-sm">
            <p className="font-medium text-blue-900">Translation Settings</p>
            <p className="text-blue-700 mt-1">
              Translating from {languages[sourceLanguage]} to {languages[targetLanguage]}
              {outputFormat === 'audio' ? ' with audio output' : ''}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LanguageSelector;