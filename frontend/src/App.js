import React, { useState, useRef } from 'react';
import { Upload, Mic, Type, Volume2, Download, Loader2 } from 'lucide-react';
import AudioUpload from './components/AudioUpload';
import AudioRecorder from './components/AudioRecorder';
import TextInput from './components/TextInput';
import LanguageSelector from './components/LanguageSelector';
import OutputDisplay from './components/OutputDisplay';
import { translateText, transcribeAudio } from './services/api';
import './App.css';

function App() {
  const [inputMode, setInputMode] = useState('text'); // 'text', 'upload', 'record'
  const [sourceLanguage, setSourceLanguage] = useState('auto');
  const [targetLanguage, setTargetLanguage] = useState('en');
  const [outputFormat, setOutputFormat] = useState('text'); // 'text', 'audio'
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleTextTranslation = async (text) => {
    if (!text.trim()) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await translateText({
        text,
        source_language: sourceLanguage,
        target_language: targetLanguage,
        output_format: outputFormat
      });
      
      setResult(response);
    } catch (err) {
      setError(err.message || 'Translation failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAudioTranscription = async (audioFile) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await transcribeAudio({
        file: audioFile,
        target_language: targetLanguage,
        output_format: outputFormat
      });
      
      setResult(response);
    } catch (err) {
      setError(err.message || 'Transcription failed');
    } finally {
      setIsLoading(false);
    }
  };

  const renderInputSection = () => {
    switch (inputMode) {
      case 'upload':
        return <AudioUpload onAudioUpload={handleAudioTranscription} disabled={isLoading} />;
      case 'record':
        return <AudioRecorder onAudioRecorded={handleAudioTranscription} disabled={isLoading} />;
      default:
        return <TextInput onTextSubmit={handleTextTranslation} disabled={isLoading} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Multimodal Translation Pipeline
          </h1>
          <p className="text-gray-600 text-lg">
            Translate speech and text between languages with AI
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          {/* Input Mode Selection */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Input Method</h2>
            <div className="flex flex-wrap gap-4">
              <button
                onClick={() => setInputMode('text')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors ${
                  inputMode === 'text'
                    ? 'bg-blue-500 text-white border-blue-500'
                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                }`}
              >
                <Type size={20} />
                Text Input
              </button>
              <button
                onClick={() => setInputMode('upload')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors ${
                  inputMode === 'upload'
                    ? 'bg-blue-500 text-white border-blue-500'
                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                }`}
              >
                <Upload size={20} />
                Upload Audio
              </button>
              <button
                onClick={() => setInputMode('record')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors ${
                  inputMode === 'record'
                    ? 'bg-blue-500 text-white border-blue-500'
                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                }`}
              >
                <Mic size={20} />
                Record Audio
              </button>
            </div>
          </div>

          {/* Language Selection */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <LanguageSelector
              sourceLanguage={sourceLanguage}
              targetLanguage={targetLanguage}
              onSourceLanguageChange={setSourceLanguage}
              onTargetLanguageChange={setTargetLanguage}
              outputFormat={outputFormat}
              onOutputFormatChange={setOutputFormat}
            />
          </div>

          {/* Input Section */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Input</h2>
            {renderInputSection()}
          </div>

          {/* Loading State */}
          {isLoading && (
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
              <div className="flex items-center justify-center gap-3">
                <Loader2 className="animate-spin" size={24} />
                <span className="text-gray-600">Processing...</span>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <div className="text-red-800">
                <strong>Error:</strong> {error}
              </div>
            </div>
          )}

          {/* Results */}
          {result && !isLoading && (
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Results</h2>
              <OutputDisplay result={result} outputFormat={outputFormat} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;