import React, { useState, useRef } from 'react';
import { Upload, FileAudio, X, AlertCircle } from 'lucide-react';

const AudioUpload = ({ onAudioUpload, disabled = false }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const supportedFormats = ['wav', 'mp3', 'flac', 'm4a', 'ogg'];
  const maxSizeMB = 50;

  const validateFile = (file) => {
    if (!file) return 'No file selected';
    
    const fileExtension = file.name.split('.').pop().toLowerCase();
    if (!supportedFormats.includes(fileExtension)) {
      return `Unsupported format. Supported formats: ${supportedFormats.join(', ')}`;
    }
    
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > maxSizeMB) {
      return `File too large. Maximum size: ${maxSizeMB}MB`;
    }
    
    return null;
  };

  const handleFileSelect = (file) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }
    
    setError(null);
    setSelectedFile(file);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = () => {
    if (selectedFile && !disabled) {
      onAudioUpload(selectedFile);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-4">
      {/* File Drop Zone */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
          dragActive
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,.mp3,.flac,.m4a,.ogg,audio/*"
          onChange={handleFileInputChange}
          className="hidden"
          disabled={disabled}
        />
        
        <div className="space-y-3">
          <Upload className="mx-auto h-12 w-12 text-gray-400" />
          <div>
            <p className="text-lg font-medium text-gray-900">
              Drop audio file here or click to browse
            </p>
            <p className="text-sm text-gray-500">
              Supported formats: {supportedFormats.join(', ')} (max {maxSizeMB}MB)
            </p>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-md">
          <AlertCircle className="h-5 w-5 text-red-400" />
          <span className="text-sm text-red-700">{error}</span>
        </div>
      )}

      {/* Selected File Display */}
      {selectedFile && !error && (
        <div className="flex items-center justify-between p-4 bg-gray-50 rounded-md">
          <div className="flex items-center gap-3">
            <FileAudio className="h-6 w-6 text-blue-500" />
            <div>
              <p className="font-medium text-gray-900">{selectedFile.name}</p>
              <p className="text-sm text-gray-500">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={handleUpload}
              disabled={disabled}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Process Audio
            </button>
            <button
              onClick={clearFile}
              disabled={disabled}
              className="p-2 text-gray-400 hover:text-gray-600 focus:outline-none"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioUpload;