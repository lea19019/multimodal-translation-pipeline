import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const translateText = async ({ text, source_language, target_language, output_format }) => {
  try {
    const response = await api.post('/translate', {
      text,
      source_language,
      target_language,
      output_format,
    });
    return response.data;
  } catch (error) {
    console.error('Translation error:', error);
    throw new Error(error.response?.data?.detail || 'Translation failed');
  }
};

export const transcribeAudio = async ({ file, target_language, output_format }) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_language', target_language);
    formData.append('output_format', output_format);

    const response = await api.post('/transcribe', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Transcription error:', error);
    throw new Error(error.response?.data?.detail || 'Transcription failed');
  }
};

export const getSupportedLanguages = async () => {
  try {
    const response = await api.get('/translation/languages');
    return response.data;
  } catch (error) {
    console.error('Error fetching languages:', error);
    throw new Error('Failed to fetch supported languages');
  }
};

export const getAudioFormats = async () => {
  try {
    const response = await api.get('/audio/formats');
    return response.data;
  } catch (error) {
    console.error('Error fetching audio formats:', error);
    throw new Error('Failed to fetch supported audio formats');
  }
};

export const getHealthStatus = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check error:', error);
    throw new Error('Health check failed');
  }
};

export default api;