import React, { useState, useRef } from 'react';
import { Volume2, Play, Pause, Download, Copy, Check } from 'lucide-react';

const OutputDisplay = ({ result, outputFormat }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [copied, setCopied] = useState(false);
  const audioRef = useRef(null);

  const handlePlayAudio = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const downloadAudio = () => {
    if (result.audio_url) {
      const link = document.createElement('a');
      link.href = result.audio_url;
      link.download = 'translated_audio.wav';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  if (!result) return null;

  return (
    <div className="space-y-6">
      {/* Original Text (if available) */}
      {result.original_text && (
        <div className="border rounded-lg p-4 bg-gray-50">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium text-gray-900">Original Text</h3>
            <button
              onClick={() => copyToClipboard(result.original_text)}
              className="p-1 text-gray-500 hover:text-gray-700 transition-colors"
              title="Copy to clipboard"
            >
              {copied ? <Check size={16} className="text-green-500" /> : <Copy size={16} />}
            </button>
          </div>
          <p className="text-gray-700">{result.original_text}</p>
        </div>
      )}

      {/* Transcribed Text (if available) */}
      {result.transcribed_text && result.transcribed_text !== result.original_text && (
        <div className="border rounded-lg p-4 bg-blue-50">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium text-blue-900">Transcribed Text</h3>
            <button
              onClick={() => copyToClipboard(result.transcribed_text)}
              className="p-1 text-blue-500 hover:text-blue-700 transition-colors"
              title="Copy to clipboard"
            >
              {copied ? <Check size={16} className="text-green-500" /> : <Copy size={16} />}
            </button>
          </div>
          <p className="text-blue-800">{result.transcribed_text}</p>
        </div>
      )}

      {/* Translated Text */}
      <div className="border rounded-lg p-4 bg-green-50">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-medium text-green-900">
            Translated Text
            {result.source_language && result.target_language && (
              <span className="text-sm font-normal text-green-700 ml-2">
                ({result.source_language} → {result.target_language})
              </span>
            )}
          </h3>
          <button
            onClick={() => copyToClipboard(result.translated_text)}
            className="p-1 text-green-500 hover:text-green-700 transition-colors"
            title="Copy to clipboard"
          >
            {copied ? <Check size={16} className="text-green-500" /> : <Copy size={16} />}
          </button>
        </div>
        <p className="text-green-800 text-lg">{result.translated_text}</p>
      </div>

      {/* Audio Output */}
      {outputFormat === 'audio' && result.audio_url && (
        <div className="border rounded-lg p-4 bg-purple-50">
          <h3 className="font-medium text-purple-900 mb-4">Audio Output</h3>
          
          <div className="flex items-center gap-4">
            <button
              onClick={handlePlayAudio}
              className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors"
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            
            <button
              onClick={downloadAudio}
              className="flex items-center gap-2 px-4 py-2 border border-purple-600 text-purple-600 rounded-md hover:bg-purple-600 hover:text-white transition-colors"
            >
              <Download size={16} />
              Download
            </button>
            
            <Volume2 className="text-purple-600" size={20} />
          </div>

          <audio
            ref={audioRef}
            src={result.audio_url}
            onEnded={handleAudioEnded}
            className="hidden"
          />
        </div>
      )}

      {/* Processing Status */}
      {result.status && (
        <div className="text-sm text-gray-500 text-center">
          Status: {result.status}
        </div>
      )}
    </div>
  );
};

export default OutputDisplay;