"""
Data loading utilities for multimodal translation evaluation.

This module provides functions to load translation samples from UUID-based
directory structures with support for different translation modalities:
- text_to_text
- audio_to_text
- text_to_audio
- audio_to_audio
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TranslationSample:
    """Represents a single translation sample with all associated data."""
    
    def __init__(
        self,
        uuid: str,
        translation_type: str,
        source_language: str,
        target_language: str,
        timestamp: str,
        source_text: Optional[str] = None,
        target_text: Optional[str] = None,
        transcribed_text: Optional[str] = None,
        source_audio_path: Optional[Path] = None,
        target_audio_path: Optional[Path] = None,
    ):
        self.uuid = uuid
        self.translation_type = translation_type
        self.source_language = source_language
        self.target_language = target_language
        self.timestamp = timestamp
        self.source_text = source_text
        self.target_text = target_text
        self.transcribed_text = transcribed_text
        self.source_audio_path = source_audio_path
        self.target_audio_path = target_audio_path
    
    def to_dict(self) -> Dict:
        """Convert sample to dictionary."""
        return {
            'uuid': self.uuid,
            'translation_type': self.translation_type,
            'source_language': self.source_language,
            'target_language': self.target_language,
            'timestamp': self.timestamp,
            'source_text': self.source_text,
            'target_text': self.target_text,
            'transcribed_text': self.transcribed_text,
            'source_audio_path': str(self.source_audio_path) if self.source_audio_path else None,
            'target_audio_path': str(self.target_audio_path) if self.target_audio_path else None,
        }


def load_sample(sample_dir: Path) -> Optional[TranslationSample]:
    """
    Load a single sample from a UUID directory.
    
    Args:
        sample_dir: Path to the UUID directory containing the sample
        
    Returns:
        TranslationSample object or None if loading fails
    """
    try:
        # Load metadata
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"No metadata.json found in {sample_dir}")
            return None
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        uuid = sample_dir.name
        translation_type = metadata.get('translation_type')
        source_language = metadata.get('source_language')
        target_language = metadata.get('target_language')
        timestamp = metadata.get('timestamp')
        
        # Load text files based on translation type
        source_text = None
        target_text = None
        transcribed_text = None
        source_audio_path = None
        target_audio_path = None
        
        # Load source text (for text_to_text and text_to_audio)
        source_txt_path = sample_dir / "source.txt"
        if source_txt_path.exists():
            with open(source_txt_path, 'r', encoding='utf-8') as f:
                source_text = f.read().strip()
        
        # Load target text (always present)
        target_txt_path = sample_dir / "target.txt"
        if target_txt_path.exists():
            with open(target_txt_path, 'r', encoding='utf-8') as f:
                target_text = f.read().strip()
        else:
            logger.warning(f"No target.txt found in {sample_dir}")
            return None
        
        # Load transcribed text (for audio_to_text and audio_to_audio)
        transcribed_txt_path = sample_dir / "transcribed.txt"
        if transcribed_txt_path.exists():
            with open(transcribed_txt_path, 'r', encoding='utf-8') as f:
                transcribed_text = f.read().strip()
        
        # Load audio files
        source_audio = sample_dir / "source_audio.wav"
        if source_audio.exists():
            source_audio_path = source_audio
        
        target_audio = sample_dir / "target_audio.wav"
        if target_audio.exists():
            target_audio_path = target_audio
        
        return TranslationSample(
            uuid=uuid,
            translation_type=translation_type,
            source_language=source_language,
            target_language=target_language,
            timestamp=timestamp,
            source_text=source_text,
            target_text=target_text,
            transcribed_text=transcribed_text,
            source_audio_path=source_audio_path,
            target_audio_path=target_audio_path,
        )
    
    except Exception as e:
        logger.error(f"Error loading sample from {sample_dir}: {e}")
        return None


def load_samples(
    data_dir: Path,
    translation_type: Optional[str] = None,
    sample_uuids: Optional[List[str]] = None,
) -> Tuple[List[TranslationSample], List[str]]:
    """
    Load all samples from a translation type directory.
    
    Args:
        data_dir: Path to the data directory (can be app_evaluation or specific type dir)
        translation_type: Optional specific translation type (text_to_text, etc.)
        sample_uuids: Optional list of specific UUIDs to load
        
    Returns:
        Tuple of (list of TranslationSample objects, list of error UUIDs)
    """
    data_dir = Path(data_dir)
    samples = []
    errors = []
    
    # Determine which directories to scan
    if translation_type:
        # Specific translation type
        type_dir = data_dir if data_dir.name == translation_type else data_dir / translation_type
        if not type_dir.exists():
            logger.error(f"Translation type directory not found: {type_dir}")
            return [], []
        scan_dirs = [type_dir]
    else:
        # Auto-detect translation type directories
        type_dirs = []
        for possible_type in ['text_to_text', 'audio_to_text', 'text_to_audio', 'audio_to_audio']:
            type_dir = data_dir / possible_type
            if type_dir.exists():
                type_dirs.append(type_dir)
        scan_dirs = type_dirs
        
        if not scan_dirs:
            logger.error(f"No translation type directories found in {data_dir}")
            return [], []
    
    # Load samples from each directory
    for scan_dir in scan_dirs:
        logger.info(f"Scanning directory: {scan_dir}")
        
        # Get all UUID directories
        uuid_dirs = [d for d in scan_dir.iterdir() if d.is_dir()]
        
        # Filter by specific UUIDs if provided
        if sample_uuids:
            uuid_dirs = [d for d in uuid_dirs if d.name in sample_uuids]
        
        logger.info(f"Found {len(uuid_dirs)} sample directories")
        
        # Load each sample
        for uuid_dir in uuid_dirs:
            sample = load_sample(uuid_dir)
            if sample:
                samples.append(sample)
            else:
                errors.append(uuid_dir.name)
    
    logger.info(f"Successfully loaded {len(samples)} samples")
    if errors:
        logger.warning(f"Failed to load {len(errors)} samples: {errors[:5]}...")
    
    return samples, errors


def get_translation_type_from_dir(data_dir: Path) -> Optional[str]:
    """
    Detect translation type from directory structure.
    
    Args:
        data_dir: Path to check
        
    Returns:
        Translation type string or None
    """
    data_dir = Path(data_dir)
    
    # Check if directory name matches a translation type
    valid_types = ['text_to_text', 'audio_to_text', 'text_to_audio', 'audio_to_audio']
    if data_dir.name in valid_types:
        return data_dir.name
    
    # Check if it contains translation type subdirectories
    for trans_type in valid_types:
        if (data_dir / trans_type).exists():
            return None  # Multiple types, can't determine single type
    
    return None


def validate_sample_for_metrics(
    sample: TranslationSample,
    required_metrics: List[str],
) -> Tuple[bool, List[str]]:
    """
    Validate that a sample has all required data for given metrics.
    
    Args:
        sample: The translation sample to validate
        required_metrics: List of metric names (e.g., ['bleu', 'mcd', 'blaser'])
        
    Returns:
        Tuple of (is_valid, list of missing components)
    """
    missing = []
    
    # Text always required
    if not sample.target_text:
        missing.append('target_text')
    
    # For text-based metrics, need source or transcribed text
    if any(m in required_metrics for m in ['bleu', 'chrf', 'comet']):
        if sample.translation_type in ['text_to_text', 'text_to_audio']:
            if not sample.source_text:
                missing.append('source_text')
        elif sample.translation_type in ['audio_to_text', 'audio_to_audio']:
            if not sample.transcribed_text:
                missing.append('transcribed_text')
    
    # For audio metrics
    if 'mcd' in required_metrics:
        if not sample.target_audio_path or not sample.target_audio_path.exists():
            missing.append('target_audio')
    
    if 'blaser' in required_metrics:
        if not sample.source_audio_path or not sample.source_audio_path.exists():
            missing.append('source_audio')
        if not sample.target_audio_path or not sample.target_audio_path.exists():
            missing.append('target_audio')
    
    return len(missing) == 0, missing
