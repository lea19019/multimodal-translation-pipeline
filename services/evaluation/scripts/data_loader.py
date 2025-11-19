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
import pandas as pd

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
        predicted_tgt_text: Optional[str] = None,
        predicted_tgt_audio_path: Optional[Path] = None,
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
        self.predicted_tgt_text = predicted_tgt_text
        self.predicted_tgt_audio_path = predicted_tgt_audio_path
    
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
            'predicted_tgt_text': self.predicted_tgt_text,
            'predicted_tgt_audio_path': str(self.predicted_tgt_audio_path) if self.predicted_tgt_audio_path else None,
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
    # For predictions mode, we use source_text even for audio_to_audio type
    if any(m in required_metrics for m in ['bleu', 'chrf', 'comet']):
        # Check if this is predictions mode (has predicted_tgt_text)
        is_predictions_mode = sample.predicted_tgt_text is not None

        if sample.translation_type in ['text_to_text', 'text_to_audio']:
            if not sample.source_text:
                missing.append('source_text')
        elif sample.translation_type in ['audio_to_text', 'audio_to_audio']:
            # For predictions mode, use source_text (from CSV src_text)
            # For ground truth mode, use transcribed_text
            if is_predictions_mode:
                if not sample.source_text:
                    missing.append('source_text')
            else:
                if not sample.transcribed_text:
                    missing.append('transcribed_text')

    # For audio metrics
    if 'mcd' in required_metrics:
        # For predictions mode, check predicted audio; for ground truth, check target audio
        is_predictions_mode = sample.predicted_tgt_audio_path is not None
        if is_predictions_mode:
            if not sample.predicted_tgt_audio_path or not sample.predicted_tgt_audio_path.exists():
                missing.append('predicted_audio')
        else:
            if not sample.target_audio_path or not sample.target_audio_path.exists():
                missing.append('target_audio')

    if 'blaser' in required_metrics:
        if not sample.source_audio_path or not sample.source_audio_path.exists():
            missing.append('source_audio')
        # For BLASER, check predicted or target audio
        is_predictions_mode = sample.predicted_tgt_audio_path is not None
        if is_predictions_mode:
            if not sample.predicted_tgt_audio_path or not sample.predicted_tgt_audio_path.exists():
                missing.append('predicted_audio')
        else:
            if not sample.target_audio_path or not sample.target_audio_path.exists():
                missing.append('target_audio')

    return len(missing) == 0, missing


def get_predicted_audio_path(
    data_dir: str,
    language: str,
    segment_id: int,
    user_id: int,
    iso_code: str
) -> Optional[Path]:
    """
    Locate predicted audio file based on segment_id and user_id.

    Args:
        data_dir: Base data directory
        language: Language name (efik, igbo, swahili, xhosa)
        segment_id: Segment identifier
        user_id: User/speaker identifier
        iso_code: ISO language code (efi, ibo, swh, xho)

    Returns:
        Path to predicted audio file or None if not found
    """
    predicted_audio_dir = Path(data_dir) / language / "predicted_tgt_audio"
    predicted_filename = f"Segment={segment_id}_User={user_id}_Language={iso_code}_pred.wav"
    predicted_path = predicted_audio_dir / predicted_filename

    if predicted_path.exists():
        return predicted_path
    else:
        return None


def load_predictions(
    data_dir: str,
    language: str,
    translation_type: str = "audio_to_audio"
) -> Tuple[List[TranslationSample], List[str]]:
    """
    Load prediction data from nmt_predictions.csv and predicted_tgt_audio/ folder.

    Returns samples with both ground truth and predicted data populated.
    Maps predictions to ground truth via segment_id + user_id.

    Args:
        data_dir: Base data directory
        language: Language name (efik, igbo, swahili, xhosa)
        translation_type: Translation type (default: audio_to_audio)

    Returns:
        Tuple of (list of TranslationSample objects, list of error messages)
    """
    data_dir_path = Path(data_dir)
    lang_dir = data_dir_path / language

    # Load NMT predictions CSV
    predictions_csv = lang_dir / "nmt_predictions.csv"
    if not predictions_csv.exists():
        logger.error(f"NMT predictions file not found: {predictions_csv}")
        return [], [f"Missing nmt_predictions.csv for {language}"]

    logger.info(f"Loading predictions from {predictions_csv}")
    df = pd.read_csv(predictions_csv, sep="|")

    samples = []
    errors = []

    # Also load ground truth CSV for audio paths
    ground_truth_csv = lang_dir / "mapped_metadata_test.csv"
    if ground_truth_csv.exists():
        df_gt = pd.read_csv(ground_truth_csv, sep="|")
        # Merge with predictions to get ground truth audio paths
        df = df.merge(df_gt[['segment_id', 'user_id', 'tgt_audio']],
                     on=['segment_id', 'user_id'],
                     how='left',
                     suffixes=('', '_gt'))

    logger.info(f"Loaded {len(df)} prediction samples for {language}")

    # Create TranslationSample objects
    for idx, row in df.iterrows():
        try:
            segment_id = row['segment_id']
            user_id = row['user_id']
            iso_code = row['iso_code']

            # Get predicted audio path
            predicted_audio_path = get_predicted_audio_path(
                data_dir, language, segment_id, user_id, iso_code
            )

            # Get ground truth audio path
            ground_truth_audio_path = None
            if 'tgt_audio' in row and pd.notna(row['tgt_audio']):
                ground_truth_audio_path = Path(row['tgt_audio'])
                if not ground_truth_audio_path.exists():
                    logger.warning(f"Ground truth audio not found: {ground_truth_audio_path}")
                    ground_truth_audio_path = None

            # Get source audio path
            src_audio_dir = lang_dir / "src_audio"
            src_audio_filename = f"Segment={segment_id}_User={user_id}_Language=en_src.wav"
            source_audio_path = src_audio_dir / src_audio_filename
            if not source_audio_path.exists():
                source_audio_path = None

            # Create sample
            sample = TranslationSample(
                uuid=f"{segment_id}_{user_id}",  # Create unique ID
                translation_type=translation_type,
                source_language="eng",
                target_language=iso_code,
                timestamp="",  # Not available in CSV
                source_text=row.get('src_text', ''),
                target_text=row.get('ground_truth_tgt_text', row.get('tgt_text', '')),
                transcribed_text=None,  # Not used for text-to-text
                source_audio_path=source_audio_path,
                target_audio_path=ground_truth_audio_path,
                predicted_tgt_text=row.get('predicted_tgt_text', ''),
                predicted_tgt_audio_path=predicted_audio_path,
            )

            samples.append(sample)

        except Exception as e:
            error_msg = f"Failed to load sample at index {idx}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    logger.info(f"Successfully created {len(samples)} samples for {language}")
    if errors:
        logger.warning(f"Encountered {len(errors)} errors")

    return samples, errors
