"""
Audio-based evaluation metrics.

Provides Mel-Cepstral Distance (MCD) computation for audio quality assessment.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AudioMetrics:
    """Compute audio-based metrics (MCD)."""
    
    def __init__(self):
        """Initialize audio metrics."""
        pass
    
    def compute_mcd(
        self,
        generated_audio_path: Union[str, Path],
        reference_audio_path: Union[str, Path],
    ) -> Dict:
        """
        Compute Mel-Cepstral Distance between generated and reference audio.
        
        Args:
            generated_audio_path: Path to generated audio file (WAV)
            reference_audio_path: Path to reference audio file (WAV)
            
        Returns:
            Dictionary with MCD score and penalty
        """
        try:
            from mel_cepstral_distance import compare_audio_files
            
            generated_audio_path = str(generated_audio_path)
            reference_audio_path = str(reference_audio_path)
            
            # Verify files exist
            if not Path(generated_audio_path).exists():
                raise FileNotFoundError(f"Generated audio not found: {generated_audio_path}")
            if not Path(reference_audio_path).exists():
                raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
            
            # Compute MCD
            logger.debug(f"Computing MCD between {generated_audio_path} and {reference_audio_path}")
            mcd, penalty = compare_audio_files(
                reference_audio_path,
                generated_audio_path,
            )
            
            return {
                'mcd': float(mcd),
                'penalty': float(penalty),
                'generated_path': generated_audio_path,
                'reference_path': reference_audio_path,
            }
        
        except ImportError as e:
            logger.error("mel-cepstral-distance package not installed")
            return {
                'mcd': -1.0,
                'penalty': -1.0,
                'error': 'Package not installed: mel-cepstral-distance',
            }
        
        except Exception as e:
            logger.error(f"Error computing MCD: {e}")
            return {
                'mcd': -1.0,
                'penalty': -1.0,
                'error': str(e),
            }
    
    def compute_mcd_batch(
        self,
        generated_audio_paths: List[Union[str, Path]],
        reference_audio_paths: List[Union[str, Path]],
    ) -> Dict:
        """
        Compute MCD for a batch of audio pairs.
        
        Args:
            generated_audio_paths: List of paths to generated audio files
            reference_audio_paths: List of paths to reference audio files
            
        Returns:
            Dictionary with scores and statistics
        """
        if len(generated_audio_paths) != len(reference_audio_paths):
            raise ValueError("Generated and reference audio lists must have same length")
        
        mcd_scores = []
        penalties = []
        errors = []
        
        for i, (gen_path, ref_path) in enumerate(zip(generated_audio_paths, reference_audio_paths)):
            result = self.compute_mcd(gen_path, ref_path)
            
            if 'error' in result:
                errors.append({'index': i, 'error': result['error']})
                logger.warning(f"Sample {i}: MCD computation failed - {result['error']}")
            else:
                mcd_scores.append(result['mcd'])
                penalties.append(result['penalty'])
        
        # Compute statistics
        if mcd_scores:
            import numpy as np
            mean_mcd = float(np.mean(mcd_scores))
            std_mcd = float(np.std(mcd_scores))
            min_mcd = float(np.min(mcd_scores))
            max_mcd = float(np.max(mcd_scores))
            mean_penalty = float(np.mean(penalties))
        else:
            mean_mcd = std_mcd = min_mcd = max_mcd = mean_penalty = -1.0
        
        return {
            'mean_mcd': mean_mcd,
            'std_mcd': std_mcd,
            'min_mcd': min_mcd,
            'max_mcd': max_mcd,
            'mean_penalty': mean_penalty,
            'mcd_scores': mcd_scores,
            'penalties': penalties,
            'num_samples': len(generated_audio_paths),
            'num_successful': len(mcd_scores),
            'num_errors': len(errors),
            'errors': errors,
        }


def compute_mcd(
    generated_audio_path: Union[str, Path],
    reference_audio_path: Union[str, Path],
) -> Dict:
    """
    Convenience function to compute MCD.
    
    Args:
        generated_audio_path: Path to generated audio
        reference_audio_path: Path to reference audio
        
    Returns:
        Dictionary with MCD score
    """
    metrics = AudioMetrics()
    return metrics.compute_mcd(generated_audio_path, reference_audio_path)


def compute_mcd_batch(
    generated_audio_paths: List[Union[str, Path]],
    reference_audio_paths: List[Union[str, Path]],
) -> Dict:
    """
    Convenience function to compute MCD for multiple pairs.
    
    Args:
        generated_audio_paths: List of generated audio paths
        reference_audio_paths: List of reference audio paths
        
    Returns:
        Dictionary with MCD scores and statistics
    """
    metrics = AudioMetrics()
    return metrics.compute_mcd_batch(generated_audio_paths, reference_audio_paths)
