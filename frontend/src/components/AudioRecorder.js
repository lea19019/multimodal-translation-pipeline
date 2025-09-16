import React, { useState, useRef } from 'react';
import { Mic, Square, Play, Pause, Download } from 'lucide-react';

const AudioRecorder = ({ onAudioRecorded, disabled = false }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioRef = useRef(null);
  const timerRef = useRef(null);

  const startRecording = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      const chunks = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        
        // Stop all tracks to release the microphone
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Failed to access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const playRecording = () => {
    if (audioRef.current) {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const pauseRecording = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      setIsPlaying(false);
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  const processRecording = () => {
    if (audioBlob && !disabled) {
      // Create a File object from the blob
      const file = new File([audioBlob], `recording-${Date.now()}.wav`, {
        type: 'audio/wav'
      });
      onAudioRecorded(file);
    }
  };

  const clearRecording = () => {
    setAudioBlob(null);
    setAudioUrl(null);
    setIsPlaying(false);
    setRecordingTime(0);
    setError(null);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-4">
      {/* Error Display */}
      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Recording Controls */}
      <div className="flex flex-col items-center space-y-4">
        {!audioBlob ? (
          <>
            {/* Record Button */}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              disabled={disabled}
              className={`w-20 h-20 rounded-full flex items-center justify-center transition-colors ${
                isRecording
                  ? 'bg-red-500 hover:bg-red-600 text-white'
                  : 'bg-blue-500 hover:bg-blue-600 text-white'
              } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {isRecording ? <Square size={32} /> : <Mic size={32} />}
            </button>
            
            {/* Recording Status */}
            <div className="text-center">
              <p className="text-lg font-medium">
                {isRecording ? 'Recording...' : 'Click to start recording'}
              </p>
              {isRecording && (
                <p className="text-2xl font-mono text-red-500 mt-2">
                  {formatTime(recordingTime)}
                </p>
              )}
            </div>
          </>
        ) : (
          <>
            {/* Playback Controls */}
            <div className="flex items-center gap-4">
              <button
                onClick={isPlaying ? pauseRecording : playRecording}
                className="w-12 h-12 rounded-full bg-blue-500 hover:bg-blue-600 text-white flex items-center justify-center"
              >
                {isPlaying ? <Pause size={20} /> : <Play size={20} />}
              </button>
              
              <div className="text-center">
                <p className="text-sm text-gray-600">Recording ready</p>
                <p className="text-lg font-mono">{formatTime(recordingTime)}</p>
              </div>
            </div>

            {/* Hidden audio element for playback */}
            <audio
              ref={audioRef}
              src={audioUrl}
              onEnded={handleAudioEnded}
              className="hidden"
            />

            {/* Action Buttons */}
            <div className="flex gap-3">
              <button
                onClick={processRecording}
                disabled={disabled}
                className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Process Recording
              </button>
              
              <button
                onClick={clearRecording}
                disabled={disabled}
                className="px-6 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Record Again
              </button>
            </div>
          </>
        )}
      </div>

      {/* Instructions */}
      <div className="text-center text-sm text-gray-500 space-y-1">
        <p>Click the microphone to start recording</p>
        <p>Click the square to stop recording</p>
        <p>Make sure your browser has microphone permissions</p>
      </div>
    </div>
  );
};

export default AudioRecorder;