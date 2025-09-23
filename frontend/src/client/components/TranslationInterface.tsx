import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, FileText, Mic, Volume2, Languages, Play, Download, Square } from 'lucide-react';
import { TranslationRequest, TranslationResponse, PipelineConfig } from '../../shared/types';

interface TranslationInterfaceProps {
  selectedPipeline: PipelineConfig | null;
  pipelines: PipelineConfig[];
  onPipelineChange: (pipeline: PipelineConfig) => void;
}

type TranslationType = 'text-to-text' | 'text-to-speech' | 'speech-to-text' | 'speech-to-speech';
type TranslationMode = 'evaluation' | 'free';

const TranslationInterface: React.FC<TranslationInterfaceProps> = ({
  selectedPipeline,
  pipelines,
  onPipelineChange
}) => {
  const [mode, setMode] = useState<TranslationMode>('free');
  const [type, setType] = useState<TranslationType>('text-to-text');
  const [sourceLang, setSourceLang] = useState('en');
  const [targetLang, setTargetLang] = useState('es');
  const [input, setInput] = useState('');
  const [reference, setReference] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<TranslationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [recordingError, setRecordingError] = useState<string>('');
  const [recordingTime, setRecordingTime] = useState(0);

  const recordingInterval = useRef<NodeJS.Timeout | null>(null);

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'it', name: 'Italian' },
    { code: 'pt', name: 'Portuguese' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' },
    { code: 'zh', name: 'Chinese' },
    { code: 'ar', name: 'Arabic' },
  ];

  const translationTypes = [
    {
      id: 'text-to-text' as const,
      label: 'Text → Text',
      icon: FileText,
      description: 'Translate text to text using NMT models'
    },
    {
      id: 'text-to-speech' as const,
      label: 'Text → Speech',
      icon: Volume2,
      description: 'Convert text to speech in target language'
    },
    {
      id: 'speech-to-text' as const,
      label: 'Speech → Text',
      icon: Mic,
      description: 'Transcribe and translate speech to text'
    },
    {
      id: 'speech-to-speech' as const,
      label: 'Speech → Speech',
      icon: Languages,
      description: 'Full speech-to-speech translation pipeline'
    }
  ];

  const handleSubmit = async () => {
    if (!selectedPipeline) {
      alert('Please select a pipeline first');
      return;
    }

    if (!input && !file && !recordedBlob) {
      alert('Please provide input text, upload a file, or record audio');
      return;
    }

    setLoading(true);
    
    try {
      const request: TranslationRequest = {
        id: `req_${Date.now()}`,
        mode,
        type,
        sourceLang,
        targetLang,
        input: input || '',
        reference: mode === 'evaluation' ? reference : undefined,
        pipelineConfig: selectedPipeline
      };

      const formData = new FormData();
      formData.append('request', JSON.stringify(request));
      
      if (file) {
        formData.append('file', file);
      } else if (recordedBlob) {
        // Convert recorded blob to file with proper naming
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const extension = recordedBlob.type.includes('webm') ? 'webm' : 'wav';
        const recordedFile = new File([recordedBlob], `recording-${timestamp}.${extension}`, { 
          type: recordedBlob.type || 'audio/webm' 
        });
        formData.append('file', recordedFile);
        console.log('Uploading recorded file:', recordedFile.name, recordedFile.size, 'bytes');
      }

      const response = await fetch('/api/translate', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      
      if (data.data) {
        setResult(data.data);
      } else {
        throw new Error(data.error || 'Translation failed');
      }
    } catch (error) {
      console.error('Translation error:', error);
      alert('Translation failed: ' + error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setInput(''); // Clear text input when file is uploaded
      setRecordedBlob(null); // Clear recorded audio
    }
  };

  const startRecording = async () => {
    try {
      // Clear any existing recording and errors
      setRecordedBlob(null);
      setFile(null);
      setInput('');
      setRecordingError('');
      setRecordingTime(0);
      
      // Check browser support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia is not supported in this browser');
      }
      
      if (!window.MediaRecorder) {
        throw new Error('MediaRecorder is not supported in this browser');
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });
      
      // Find supported MIME type
      const mimeTypes = ['audio/webm', 'audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg'];
      const supportedMimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type));
      
      const recorder = new MediaRecorder(stream, supportedMimeType ? { mimeType: supportedMimeType } : {});
      const chunks: BlobPart[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      recorder.onstop = () => {
        const mimeType = recorder.mimeType || 'audio/webm';
        const blob = new Blob(chunks, { type: mimeType });
        setRecordedBlob(blob);
        setFile(null);
        setInput('');
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
        
        setIsRecording(false);
        if (recordingInterval.current) {
          clearInterval(recordingInterval.current);
          recordingInterval.current = null;
        }
      };

      recorder.onerror = (e) => {
        console.error('Recording error:', e);
        setRecordingError('Recording failed: ' + (e.error || 'Unknown error'));
        setIsRecording(false);
        if (recordingInterval.current) {
          clearInterval(recordingInterval.current);
          recordingInterval.current = null;
        }
      };

      recorder.start(1000);
      setMediaRecorder(recorder);
      setIsRecording(true);

      // Start timer
      recordingInterval.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (error) {
      console.error('Error in startRecording:', error);
      setIsRecording(false);
      
      if (error instanceof Error) {
        if (error.name === 'NotAllowedError') {
          setRecordingError('Microphone access denied. Please allow microphone permission and try again.');
        } else if (error.name === 'NotFoundError') {
          setRecordingError('No microphone found. Please connect a microphone and try again.');
        } else if (error.name === 'NotSupportedError') {
          setRecordingError('MediaRecorder is not supported in this browser.');
        } else {
          setRecordingError(`Error accessing microphone: ${error.message}`);
        }
      } else {
        setRecordingError('Error accessing microphone. Please ensure you have granted permission.');
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    } else {
      setIsRecording(false);
      if (recordingInterval.current) {
        clearInterval(recordingInterval.current);
        recordingInterval.current = null;
      }
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const requiresAudioInput = type.startsWith('speech-');

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      if (recordingInterval.current) {
        clearInterval(recordingInterval.current);
      }
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
    };
  }, [mediaRecorder]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Translation Interface</h2>
          <p className="text-slate-400 mt-1">
            Test your multilingual translation pipelines with various input types
          </p>
        </div>
        
        {selectedPipeline && (
          <div className="card p-4">
            <div className="text-sm font-medium text-slate-200">Active Pipeline</div>
            <div className="text-lg font-bold text-blue-400">{selectedPipeline.name}</div>
            <div className="text-xs text-slate-400">{selectedPipeline.description}</div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration Panel */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold text-white">Configuration</h3>
          </div>
          <div className="card-content space-y-4">
            {/* Mode Selection */}
            <div className="form-group">
              <label className="label">Translation Mode</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setMode('free')}
                  className={`p-3 rounded-md border text-sm font-medium transition-colors ${
                    mode === 'free'
                      ? 'bg-blue-600 border-blue-500 text-white'
                      : 'bg-dark-700 border-dark-600 text-slate-300 hover:bg-dark-600'
                  }`}
                >
                  Free Translation
                </button>
                <button
                  onClick={() => setMode('evaluation')}
                  className={`p-3 rounded-md border text-sm font-medium transition-colors ${
                    mode === 'evaluation'
                      ? 'bg-blue-600 border-blue-500 text-white'
                      : 'bg-dark-700 border-dark-600 text-slate-300 hover:bg-dark-600'
                  }`}
                >
                  Evaluation Mode
                </button>
              </div>
            </div>

            {/* Pipeline Selection */}
            <div className="form-group">
              <label className="label">Pipeline</label>
              <select
                value={selectedPipeline?.id || ''}
                onChange={(e) => {
                  const pipeline = pipelines.find(p => p.id === e.target.value);
                  if (pipeline) onPipelineChange(pipeline);
                }}
                className="select"
              >
                <option value="">Select a pipeline...</option>
                {pipelines.map(pipeline => (
                  <option key={pipeline.id} value={pipeline.id}>
                    {pipeline.name} - {pipeline.description}
                  </option>
                ))}
              </select>
            </div>

            {/* Translation Type */}
            <div className="form-group">
              <label className="label">Translation Type</label>
              <div className="grid grid-cols-1 gap-2">
                {translationTypes.map(translationType => {
                  const Icon = translationType.icon;
                  return (
                    <button
                      key={translationType.id}
                      onClick={() => setType(translationType.id)}
                      className={`flex items-center space-x-3 p-3 rounded-md border text-left transition-colors ${
                        type === translationType.id
                          ? 'bg-blue-600 border-blue-500 text-white'
                          : 'bg-dark-700 border-dark-600 text-slate-300 hover:bg-dark-600'
                      }`}
                    >
                      <Icon className="h-5 w-5" />
                      <div>
                        <div className="font-medium">{translationType.label}</div>
                        <div className="text-xs opacity-80">{translationType.description}</div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Language Selection */}
            <div className="grid grid-cols-2 gap-4">
              <div className="form-group">
                <label className="label">Source Language</label>
                <select
                  value={sourceLang}
                  onChange={(e) => setSourceLang(e.target.value)}
                  className="select"
                >
                  {languages.map(lang => (
                    <option key={lang.code} value={lang.code}>
                      {lang.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label className="label">Target Language</label>
                <select
                  value={targetLang}
                  onChange={(e) => setTargetLang(e.target.value)}
                  className="select"
                >
                  {languages.map(lang => (
                    <option key={lang.code} value={lang.code}>
                      {lang.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Input Panel */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold text-white">Input</h3>
          </div>
          <div className="card-content space-y-4">
            {/* Input Method Selector - Only for non-speech inputs */}
            {!requiresAudioInput && (
              <div className="form-group">
                <label className="label">Input Method</label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => {setFile(null); setRecordedBlob(null); setInput('');}}
                    className={`btn ${!file && !recordedBlob ? 'btn-primary' : 'btn-secondary'}`}
                  >
                    <FileText className="h-4 w-4 mr-2" />
                    Text Input
                  </button>
                  <button
                    onClick={() => setInput('')}
                    className={`btn ${file || recordedBlob ? 'btn-primary' : 'btn-secondary'}`}
                  >
                    <Upload className="h-4 w-4 mr-2" />
                    File Upload
                  </button>
                </div>
              </div>
            )}

            {/* Speech Input Methods */}
            {requiresAudioInput && (
              <div className="form-group">
                <label className="label">Audio Input Method</label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => {setFile(null); setRecordedBlob(null);}}
                    className={`btn ${!file && !recordedBlob ? 'btn-primary' : 'btn-secondary'}`}
                  >
                    <Mic className="h-4 w-4 mr-2" />
                    Record Live
                  </button>
                  <button
                    onClick={() => setRecordedBlob(null)}
                    className={`btn ${file ? 'btn-primary' : 'btn-secondary'}`}
                  >
                    <Upload className="h-4 w-4 mr-2" />
                    Upload File
                  </button>
                </div>
              </div>
            )}

            {/* Live Recording Interface */}
            {requiresAudioInput && !file && (
              <div className="form-group">
                <label className="label">Live Recording</label>
                <div className="bg-dark-900 border border-dark-600 rounded-md p-6">
                  {recordingError && (
                    <div className="mb-4 p-3 bg-red-900 border border-red-600 rounded text-red-300 text-sm">
                      {recordingError}
                      <button 
                        onClick={() => setRecordingError('')}
                        className="ml-2 text-red-400 hover:text-red-300"
                      >
                        ✕
                      </button>
                    </div>
                  )}
                  
                  {!isRecording && !recordedBlob && (
                    <div className="text-center">
                      <button
                        onClick={startRecording}
                        className="btn-primary mb-4"
                      >
                        <Mic className="h-5 w-5 mr-2" />
                        Start Recording
                      </button>
                      <p className="text-sm text-slate-400">
                        Click to start recording your voice
                      </p>
                    </div>
                  )}

                  {isRecording && (
                    <div className="text-center">
                      <div className="flex items-center justify-center mb-4">
                        <div className="w-4 h-4 bg-red-500 rounded-full animate-pulse mr-3"></div>
                        <span className="text-red-400 font-medium">Recording...</span>
                      </div>
                      <div className="text-2xl font-mono text-white mb-4">
                        {formatTime(recordingTime)}
                      </div>
                      <button
                        onClick={stopRecording}
                        className="btn-danger"
                      >
                        <Square className="h-4 w-4 mr-2" />
                        Stop Recording
                      </button>
                    </div>
                  )}

                  {recordedBlob && !isRecording && (
                    <div className="text-center">
                      <div className="text-green-400 mb-4">
                        ✓ Recording completed ({formatTime(recordingTime)})
                      </div>
                      <div className="flex justify-center space-x-2">
                        <button
                          onClick={startRecording}
                          className="btn-secondary"
                        >
                          <Mic className="h-4 w-4 mr-2" />
                          Record Again
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Text Input - Only for non-speech inputs */}
            {!requiresAudioInput && !file && !recordedBlob && (
              <div className="form-group">
                <label className="label">
                  {type === 'text-to-text' ? `${sourceLang.toUpperCase()} Text` : `${sourceLang.toUpperCase()} Text to Convert to Speech`}
                </label>
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={
                    type === 'text-to-text' ? `Enter ${sourceLang.toUpperCase()} text to translate...` :
                    `Enter ${sourceLang.toUpperCase()} text to convert to speech...`
                  }
                  className="textarea h-32"
                />
              </div>
            )}

            {/* File Upload */}
            {(file || (!input.trim() && !recordedBlob && !requiresAudioInput) || (requiresAudioInput && !recordedBlob)) && (
              <div className="form-group">
                <label className="label">
                  {requiresAudioInput ? 'Audio File' : 'Text File'}
                </label>
                <div className="file-upload-zone">
                  <input
                    type="file"
                    accept={requiresAudioInput ? '.wav,.mp3,.m4a,.flac' : '.txt,.json'}
                    onChange={handleFileUpload}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <Upload className="h-8 w-8 text-slate-400 mx-auto mb-2" />
                    <p className="text-slate-300">
                      {file ? file.name : `Click to upload ${requiresAudioInput ? 'audio' : 'text'} file`}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      {requiresAudioInput ? 
                        'Supports WAV, MP3, M4A, FLAC (max 100MB)' : 
                        'Supports TXT, JSON files (max 100MB)'
                      }
                    </p>
                  </label>
                </div>
              </div>
            )}

            {/* Reference (Evaluation Mode) */}
            {mode === 'evaluation' && (
              <div className="form-group">
                <label className="label">Reference Translation</label>
                <textarea
                  value={reference}
                  onChange={(e) => setReference(e.target.value)}
                  placeholder={`Enter expected ${targetLang.toUpperCase()} translation...`}
                  className="textarea h-24"
                />
              </div>
            )}

            {/* Submit Button */}
            <button
              onClick={handleSubmit}
              disabled={loading || !selectedPipeline}
              className="btn-primary w-full flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <div className="loading-spinner h-4 w-4"></div>
                  <span>Translating...</span>
                </>
              ) : (
                <>
                  <Send className="h-4 w-4" />
                  <span>Translate</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-semibold text-white">Translation Results</h3>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Output */}
              <div>
                <label className="label">Output</label>
                {(type === 'text-to-speech' || type === 'speech-to-speech') ? (
                  <div className="bg-dark-900 border border-dark-600 rounded-md p-4">
                    <div className="flex items-center space-x-4">
                      <Play className="h-6 w-6 text-blue-400" />
                      <div>
                        <div className="text-sm font-medium">Audio Output</div>
                        <div className="text-xs text-slate-400">Generated speech file</div>
                      </div>
                      <button className="btn-secondary ml-auto">
                        <Download className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="bg-dark-900 border border-dark-600 rounded-md p-4">
                    <p className="text-slate-100">
                      {result.result?.translatedText || result.output || 'No translation result'}
                    </p>
                  </div>
                )}
              </div>

              {/* Metrics */}
              {result.evaluationMetrics && (
                <div>
                  <label className="label">Evaluation Metrics</label>
                  <div className="grid grid-cols-2 gap-3">
                    {result.evaluationMetrics.bleu && (
                      <div className="metric-card">
                        <div className="metric-value">{result.evaluationMetrics.bleu.toFixed(2)}</div>
                        <div className="metric-label">BLEU Score</div>
                      </div>
                    )}
                    {result.evaluationMetrics.comet && (
                      <div className="metric-card">
                        <div className="metric-value">{result.evaluationMetrics.comet.toFixed(3)}</div>
                        <div className="metric-label">COMET Score</div>
                      </div>
                    )}
                    {result.evaluationMetrics.wer && (
                      <div className="metric-card">
                        <div className="metric-value">{(result.evaluationMetrics.wer * 100).toFixed(1)}%</div>
                        <div className="metric-label">Word Error Rate</div>
                      </div>
                    )}
                    {result.evaluationMetrics.mos && (
                      <div className="metric-card">
                        <div className="metric-value">{result.evaluationMetrics.mos.toFixed(2)}</div>
                        <div className="metric-label">MOS</div>
                      </div>
                    )}
                  </div>
                  
                  <div className="mt-4 text-sm text-slate-400">
                    Processing time: {result.processingTime?.toFixed(0) || 'N/A'}ms
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TranslationInterface;