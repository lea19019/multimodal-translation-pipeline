"""
Data Logger for Evaluation System

Saves translation results with metadata to app_evaluation folders for later evaluation.
Supports all translation modes: text-to-text, text-to-audio, audio-to-text, audio-to-audio.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import base64
import wave
import io
import struct


class EvaluationDataLogger:
    """Handles saving translation data for evaluation purposes."""
    
    def __init__(self, base_data_dir: str = None):
        """
        Initialize the data logger.
        
        Args:
            base_data_dir: Base directory for evaluation data. 
                          Defaults to services/data/app_evaluation
        """
        if base_data_dir is None:
            # Default to services/data/app_evaluation
            current_dir = Path(__file__).parent
            services_dir = current_dir.parent
            base_data_dir = services_dir / "data" / "app_evaluation"
        
        self.base_dir = Path(base_data_dir)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create all required directories if they don't exist."""
        task_types = ["text_to_text", "text_to_audio", "audio_to_text", "audio_to_audio"]
        for task_type in task_types:
            (self.base_dir / task_type).mkdir(parents=True, exist_ok=True)
    
    def _convert_to_wav(self, audio_bytes: bytes, sample_rate: int = 22050) -> bytes:
        """
        Convert raw PCM audio bytes to WAV format.
        
        The audio services may send raw float32 PCM data. This function
        detects the format and converts to proper WAV if needed.
        
        Args:
            audio_bytes: Raw audio bytes (could be WAV or raw PCM)
            sample_rate: Sample rate for raw PCM (default 22050 for TTS, 16000 for ASR)
        
        Returns:
            Proper WAV file bytes
        """
        # Check if it's already a WAV file
        if audio_bytes[:4] == b'RIFF':
            return audio_bytes
        
        # Otherwise, assume it's raw float32 PCM and convert to WAV
        try:
            # Calculate number of float32 samples
            num_samples = len(audio_bytes) // 4  # 4 bytes per float32
            
            # Convert bytes to float32 values and then to int16
            int16_samples = []
            for i in range(num_samples):
                # Extract 4 bytes and unpack as float32
                float_bytes = audio_bytes[i*4:(i+1)*4]
                float_value = struct.unpack('<f', float_bytes)[0]  # '<f' = little-endian float
                
                # Convert float [-1.0, 1.0] to int16 [-32768, 32767]
                # Clamp to prevent overflow
                float_value = max(-1.0, min(1.0, float_value))
                int16_value = int(float_value * 32767)
                int16_samples.append(int16_value)
            
            # Pack int16 samples back to bytes
            int16_bytes = struct.pack(f'<{len(int16_samples)}h', *int16_samples)  # '<h' = little-endian short
            
            # Create WAV file in memory
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(int16_bytes)
            
            return wav_io.getvalue()
        except Exception as e:
            # If conversion fails, return original bytes
            # (might be a different format)
            return audio_bytes
    
    
    def save_translation(
        self,
        task_type: str,
        source_text: Optional[str],
        target_text: Optional[str],
        source_lang: str,
        target_lang: str,
        source_audio: Optional[bytes] = None,
        target_audio: Optional[bytes] = None,
        transcribed_text: Optional[str] = None,
        models_used: Optional[Dict[str, str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save translation data with metadata.
        
        Args:
            task_type: One of: text_to_text, text_to_audio, audio_to_text, audio_to_audio
            source_text: Original input text (if text input)
            target_text: Translated/output text
            source_lang: Source language code (e.g., 'en', 'es')
            target_lang: Target language code
            source_audio: Original input audio bytes (if audio input)
            target_audio: Generated output audio bytes (if audio output)
            transcribed_text: ASR output (if audio input)
            models_used: Dict with keys: asr_model, nmt_model, tts_model
            additional_metadata: Any extra metadata to store
        
        Returns:
            The unique ID assigned to this sample
        """
        # Generate unique ID
        sample_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Create directory for this task type
        task_dir = self.base_dir / task_type
        sample_dir = task_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Build metadata
        metadata = {
            "id": sample_id,
            "timestamp": timestamp,
            "task_type": task_type,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "language_pair": f"{source_lang}-{target_lang}",
            "models": models_used or {},
            "files": {}
        }
        
        # Save text files
        if source_text is not None:
            source_text_path = sample_dir / "source.txt"
            source_text_path.write_text(source_text, encoding="utf-8")
            metadata["files"]["source_text"] = "source.txt"
            metadata["source_text"] = source_text
        
        if transcribed_text is not None:
            transcribed_path = sample_dir / "transcribed.txt"
            transcribed_path.write_text(transcribed_text, encoding="utf-8")
            metadata["files"]["transcribed_text"] = "transcribed.txt"
            metadata["transcribed_text"] = transcribed_text
        
        if target_text is not None:
            target_text_path = sample_dir / "target.txt"
            target_text_path.write_text(target_text, encoding="utf-8")
            metadata["files"]["target_text"] = "target.txt"
            metadata["target_text"] = target_text
        
        # Save audio files
        if source_audio is not None:
            # Convert to WAV format if needed (ASR typically uses 16kHz)
            source_audio_wav = self._convert_to_wav(source_audio, sample_rate=16000)
            source_audio_path = sample_dir / "source_audio.wav"
            source_audio_path.write_bytes(source_audio_wav)
            metadata["files"]["source_audio"] = "source_audio.wav"
            metadata["source_audio_info"] = self._get_audio_info(source_audio_wav)
        
        if target_audio is not None:
            # Convert to WAV format if needed (TTS typically uses 22050Hz)
            target_audio_wav = self._convert_to_wav(target_audio, sample_rate=22050)
            target_audio_path = sample_dir / "target_audio.wav"
            target_audio_path.write_bytes(target_audio_wav)
            metadata["files"]["target_audio"] = "target_audio.wav"
            metadata["target_audio_info"] = self._get_audio_info(target_audio_wav)
        
        # Add additional metadata
        if additional_metadata:
            metadata["additional"] = additional_metadata
        
        # Save metadata JSON
        metadata_path = sample_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, indent=2, fp=f)
        
        return sample_id
    
    def _get_audio_info(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Extract basic info from audio bytes."""
        try:
            with io.BytesIO(audio_bytes) as audio_io:
                with wave.open(audio_io, 'rb') as wav:
                    return {
                        "channels": wav.getnchannels(),
                        "sample_width": wav.getsampwidth(),
                        "framerate": wav.getframerate(),
                        "frames": wav.getnframes(),
                        "duration_seconds": wav.getnframes() / wav.getframerate()
                    }
        except Exception as e:
            return {"error": str(e)}
    
    def get_task_samples(self, task_type: str, language_pair: Optional[str] = None) -> list:
        """
        Get all samples for a specific task type.
        
        Args:
            task_type: The task type to query
            language_pair: Optional filter by language pair (e.g., 'en-es')
        
        Returns:
            List of metadata dictionaries
        """
        task_dir = self.base_dir / task_type
        if not task_dir.exists():
            return []
        
        samples = []
        for sample_dir in task_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            
            metadata_path = sample_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Filter by language pair if specified
            if language_pair and metadata.get("language_pair") != language_pair:
                continue
            
            # Add full paths to files
            metadata["sample_dir"] = str(sample_dir.absolute())
            samples.append(metadata)
        
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored evaluation data."""
        stats = {
            "total_samples": 0,
            "by_task_type": {},
            "by_language_pair": {}
        }
        
        for task_type in ["text_to_text", "text_to_audio", "audio_to_text", "audio_to_audio"]:
            samples = self.get_task_samples(task_type)
            count = len(samples)
            stats["by_task_type"][task_type] = count
            stats["total_samples"] += count
            
            # Count by language pair
            for sample in samples:
                lang_pair = sample.get("language_pair", "unknown")
                stats["by_language_pair"][lang_pair] = stats["by_language_pair"].get(lang_pair, 0) + 1
        
        return stats
