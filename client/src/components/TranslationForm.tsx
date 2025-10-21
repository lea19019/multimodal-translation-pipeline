'use client';

import { useState, useEffect } from 'react';
import { 
  ArrowRight, 
  Loader2, 
  Play, 
  Pause, 
  Mic, 
  StopCircle,
  Volume2,
  AlertCircle 
} from 'lucide-react';
import { MultimodalTranslationClient, SUPPORTED_LANGUAGES, AudioUtils } from '@/lib/api-client';
import { useAudioRecorder } from '@/hooks/useAudioRecorder';

interface TranslationFormProps {
  mode: 'text' | 'audio-to-text' | 'text-to-audio' | 'audio-to-audio';
}

export default function TranslationForm({ mode }: TranslationFormProps) {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [outputAudio, setOutputAudio] = useState<string | null>(null);
  const [sourceLanguage, setSourceLanguage] = useState('en');
  const [targetLanguage, setTargetLanguage] = useState('es');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [isPlayingRecording, setIsPlayingRecording] = useState(false);
  
  // Intermediate results from translation pipeline
  const [transcribedText, setTranscribedText] = useState<string | null>(null);
  const [translatedText, setTranslatedText] = useState<string | null>(null);

  // Model selection state
  const [asrModel, setAsrModel] = useState('base');
  const [nmtModel, setNmtModel] = useState('base');
  const [ttsModel, setTtsModel] = useState('base');
  const [availableAsrModels, setAvailableAsrModels] = useState<string[]>(['base']);
  const [availableNmtModels, setAvailableNmtModels] = useState<string[]>(['base']);
  const [availableTtsModels, setAvailableTtsModels] = useState<string[]>(['base']);

  // Evaluation mode state
  const [saveForEvaluation, setSaveForEvaluation] = useState(false);

  const audioRecorder = useAudioRecorder();

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const client = new MultimodalTranslationClient(
          process.env.NEXT_PUBLIC_GATEWAY_URL,
          process.env.NEXT_PUBLIC_ASR_URL,
          process.env.NEXT_PUBLIC_NMT_URL,
          process.env.NEXT_PUBLIC_TTS_URL
        );

        // Fetch models from gateway (falls back to direct service calls if gateway unavailable)
        const [asrModels, nmtModels, ttsModels] = await Promise.all([
          client.getASRModelsFromGateway().catch(() => client.getASRModels()).catch(() => ['base']),
          client.getNMTModelsFromGateway().catch(() => client.getNMTModels()).catch(() => ['base']),
          client.getTTSModelsFromGateway().catch(() => client.getTTSModels()).catch(() => ['base']),
        ]);

        setAvailableAsrModels(asrModels.length > 0 ? asrModels : ['base']);
        setAvailableNmtModels(nmtModels.length > 0 ? nmtModels : ['base']);
        setAvailableTtsModels(ttsModels.length > 0 ? ttsModels : ['base']);
      } catch (error) {
        console.error('Error fetching models:', error);
        // Keep default 'base' models if fetching fails
      }
    };

    fetchModels();
  }, []);

  const handleTranslate = async () => {
    setLoading(true);
    setError(null);
    setOutputText('');
    setOutputAudio(null);
    setTranscribedText(null);
    setTranslatedText(null);

    try {
      const client = new MultimodalTranslationClient(
        process.env.NEXT_PUBLIC_GATEWAY_URL,
        process.env.NEXT_PUBLIC_ASR_URL,
        process.env.NEXT_PUBLIC_NMT_URL,
        process.env.NEXT_PUBLIC_TTS_URL
      );

      let inputData: string;
      let inputType: 'text' | 'audio';
      let outputType: 'text' | 'audio';

      // Determine input type and prepare input data
      if (mode === 'audio-to-text' || mode === 'audio-to-audio') {
        if (!audioRecorder.audioData) {
          throw new Error('No audio recorded');
        }
        inputData = AudioUtils.float32ArrayToBase64(audioRecorder.audioData);
        inputType = 'audio';
      } else {
        inputData = inputText;
        inputType = 'text';
      }

      // Determine output type
      outputType = (mode === 'text-to-audio' || mode === 'audio-to-audio') ? 'audio' : 'text';

      // Call the translate endpoint which returns intermediate results
      const result = await client.translate({
        input: inputData,
        input_type: inputType,
        source_language: sourceLanguage,
        target_language: targetLanguage,
        output_type: outputType,
        asr_model: asrModel,
        nmt_model: nmtModel,
        tts_model: ttsModel,
        save_for_evaluation: saveForEvaluation,
      });

      // Store intermediate results
      if (result.transcribed_text) {
        setTranscribedText(result.transcribed_text);
      }
      if (result.translated_text) {
        setTranslatedText(result.translated_text);
      }

      // Set final output
      if (result.output_type === 'audio') {
        setOutputAudio(result.output);
        setOutputText(''); // Clear text output for audio mode
      } else {
        setOutputText(result.output);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Translation failed');
    } finally {
      setLoading(false);
    }
  };

  const playAudio = async () => {
    if (!outputAudio) return;
    setIsPlayingAudio(true);
    try {
      await AudioUtils.playAudioFromBase64(outputAudio);
    } catch (err) {
      setError('Failed to play audio');
    } finally {
      setIsPlayingAudio(false);
    }
  };

  const playRecording = async () => {
    if (!audioRecorder.audioBlob) return;
    setIsPlayingRecording(true);
    try {
      const url = URL.createObjectURL(audioRecorder.audioBlob);
      const audio = new Audio(url);
      audio.onended = () => {
        setIsPlayingRecording(false);
        URL.revokeObjectURL(url);
      };
      audio.onerror = () => {
        setIsPlayingRecording(false);
        URL.revokeObjectURL(url);
        setError('Failed to play recording');
      };
      await audio.play();
    } catch (err) {
      setIsPlayingRecording(false);
      setError('Failed to play recording');
    }
  };

  const needsAudioInput = mode === 'audio-to-text' || mode === 'audio-to-audio';
  const needsTextInput = mode === 'text' || mode === 'text-to-audio';
  const producesAudio = mode === 'text-to-audio' || mode === 'audio-to-audio';

  const canTranslate = needsTextInput 
    ? inputText.trim().length > 0 
    : audioRecorder.audioData !== null;

  return (
    <div className="max-w-4xl mx-auto bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      {/* Language Selection */}
      <div className="flex items-center gap-4 mb-6">
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Source Language
          </label>
          <select
            value={sourceLanguage}
            onChange={(e) => setSourceLanguage(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            {SUPPORTED_LANGUAGES.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
          </select>
        </div>

        <ArrowRight className="w-6 h-6 text-gray-400 mt-6" />

        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Target Language
          </label>
          <select
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            {SUPPORTED_LANGUAGES.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Model Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* ASR Model Selection (only for audio input modes) */}
        {needsAudioInput && (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              ASR Model
            </label>
            <select
              value={asrModel}
              onChange={(e) => setAsrModel(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              {availableAsrModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* NMT Model Selection (always shown) */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            NMT Model
          </label>
          <select
            value={nmtModel}
            onChange={(e) => setNmtModel(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            {availableNmtModels.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>

        {/* TTS Model Selection (only for audio output modes) */}
        {producesAudio && (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              TTS Model
            </label>
            <select
              value={ttsModel}
              onChange={(e) => setTtsModel(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              {availableTtsModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Evaluation Mode Checkbox */}
      <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={saveForEvaluation}
            onChange={(e) => setSaveForEvaluation(e.target.checked)}
            className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
          />
          <div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              Save for Evaluation
            </span>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
              Save this translation data for later evaluation with BLEU, COMET, chrF++, MCD, BLASER 2.0, and MOS metrics
            </p>
          </div>
        </label>
      </div>

      {/* Input Area */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Input
        </label>

        {needsTextInput && (
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to translate..."
            rows={6}
            className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
          />
        )}

        {needsAudioInput && (
          <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8">
            <div className="flex flex-col items-center justify-center gap-4">
              {!audioRecorder.isRecording && !audioRecorder.audioData && (
                <>
                  <Mic className="w-12 h-12 text-gray-400" />
                  <p className="text-gray-600 dark:text-gray-400 text-center">
                    Click the button below to start recording
                  </p>
                  <button
                    onClick={audioRecorder.startRecording}
                    className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2"
                  >
                    <Mic className="w-5 h-5" />
                    Start Recording
                  </button>
                </>
              )}

              {audioRecorder.isRecording && (
                <>
                  <div className="relative">
                    <Mic className="w-12 h-12 text-red-600 animate-pulse" />
                    <div className="absolute -inset-2 border-4 border-red-600 rounded-full animate-ping opacity-75" />
                  </div>
                  <p className="text-red-600 font-medium">Recording...</p>
                  <button
                    onClick={audioRecorder.stopRecording}
                    className="px-6 py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-900 transition-colors flex items-center gap-2"
                  >
                    <StopCircle className="w-5 h-5" />
                    Stop Recording
                  </button>
                </>
              )}

              {!audioRecorder.isRecording && audioRecorder.audioData && (
                <>
                  <Volume2 className="w-12 h-12 text-green-600" />
                  <p className="text-green-600 font-medium">Audio recorded successfully</p>
                  <div className="flex gap-2">
                    <button
                      onClick={playRecording}
                      disabled={isPlayingRecording}
                      className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                    >
                      <Volume2 className="w-4 h-4" />
                      {isPlayingRecording ? 'Playing...' : 'Play'}
                    </button>
                    <button
                      onClick={audioRecorder.clearRecording}
                      className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                    >
                      Clear
                    </button>
                    <button
                      onClick={audioRecorder.startRecording}
                      className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2"
                    >
                      <Mic className="w-4 h-4" />
                      Record Again
                    </button>
                  </div>
                </>
              )}

              {audioRecorder.error && (
                <div className="flex items-center gap-2 text-red-600">
                  <AlertCircle className="w-5 h-5" />
                  <span className="text-sm">{audioRecorder.error}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Translate Button */}
      <button
        onClick={handleTranslate}
        disabled={loading || !canTranslate}
        className="w-full py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
      >
        {loading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Translating...
          </>
        ) : (
          <>
            <ArrowRight className="w-5 h-5" />
            Translate
          </>
        )}
      </button>

      {/* Error Display */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-center gap-2 text-red-800 dark:text-red-200">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Output Area with Intermediate Results */}
      {(transcribedText || translatedText || outputText || outputAudio) && (
        <div className="mt-6 space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Translation Results
          </h3>

          {/* Step 1: Transcription (if audio input) */}
          {transcribedText && (
            <div className="border border-blue-200 dark:border-blue-800 rounded-lg overflow-hidden">
              <div className="bg-blue-50 dark:bg-blue-900/20 px-4 py-2 border-b border-blue-200 dark:border-blue-800">
                <div className="flex items-center gap-2">
                  <Volume2 className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  <span className="text-sm font-medium text-blue-900 dark:text-blue-100">
                    Step 1: Transcription ({sourceLanguage})
                  </span>
                </div>
              </div>
              <div className="p-4 bg-white dark:bg-gray-800">
                <p className="text-gray-900 dark:text-white whitespace-pre-wrap">
                  {transcribedText}
                </p>
              </div>
            </div>
          )}

          {/* Step 2: Translation (always shown) */}
          {translatedText && (
            <div className="border border-green-200 dark:border-green-800 rounded-lg overflow-hidden">
              <div className="bg-green-50 dark:bg-green-900/20 px-4 py-2 border-b border-green-200 dark:border-green-800">
                <div className="flex items-center gap-2">
                  <ArrowRight className="w-4 h-4 text-green-600 dark:text-green-400" />
                  <span className="text-sm font-medium text-green-900 dark:text-green-100">
                    Step 2: Translation ({sourceLanguage} → {targetLanguage})
                  </span>
                </div>
              </div>
              <div className="p-4 bg-white dark:bg-gray-800">
                <p className="text-gray-900 dark:text-white whitespace-pre-wrap">
                  {translatedText}
                </p>
              </div>
            </div>
          )}

          {/* Step 3: Audio Output (if audio output) */}
          {producesAudio && outputAudio && (
            <div className="border border-purple-200 dark:border-purple-800 rounded-lg overflow-hidden">
              <div className="bg-purple-50 dark:bg-purple-900/20 px-4 py-2 border-b border-purple-200 dark:border-purple-800">
                <div className="flex items-center gap-2">
                  <Volume2 className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                  <span className="text-sm font-medium text-purple-900 dark:text-purple-100">
                    Step 3: Audio Synthesis ({targetLanguage})
                  </span>
                </div>
              </div>
              <div className="p-4 bg-white dark:bg-gray-800 flex justify-center">
                <button
                  onClick={playAudio}
                  disabled={isPlayingAudio}
                  className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  {isPlayingAudio ? (
                    <>
                      <Pause className="w-5 h-5" />
                      Playing...
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      Play Audio
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Fallback for text-only output (shouldn't happen with new API) */}
          {outputText && !transcribedText && !translatedText && (
            <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600">
              <p className="text-gray-900 dark:text-white whitespace-pre-wrap">
                {outputText}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
