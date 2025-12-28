"""
COMET evaluation module.

Provides COMET quality estimation for machine translation using
the Unbabel COMET model (wmt22-comet-da by default).
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class CometEvaluator:
    """Evaluate translation quality using COMET metric."""
    
    def __init__(
        self,
        model_name: str = "McGill-NLP/ssa-comet-qe",
        device: Optional[torch.device] = None,
        batch_size: int = 8,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize COMET evaluator.
        
        Args:
            model_name: Hugging Face model name (default: wmt22-comet-da)
            device: Torch device (auto-detect if None)
            batch_size: Batch size for evaluation
            cache_dir: Directory to cache models (default: ./models/comet/)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Set cache directory
        if cache_dir is None:
            # Put models in services/evaluation/comet/models/
            cache_dir = Path(__file__).parent.parent / "comet" / "models"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model lazily
        self.model = None
        self._model_loaded = False
    
    def _load_model(self):
        """Load the COMET model (called on first use)."""
        if self._model_loaded:
            return

        try:
            from comet import download_model, load_from_checkpoint

            logger.info(f"Loading COMET model: {self.model_name}")
            
            # Offline mode should already be set globally, but ensure it here too
            # Try to download model first (will use local cache in offline mode)
            try:
                model_path = download_model(self.model_name, saving_directory=str(self.cache_dir))
            except Exception as download_error:
                logger.warning(f"download_model failed: {download_error}")

                # Fall back to loading from local checkpoint if it exists
                if self.model_name == "McGill-NLP/ssa-comet-qe":
                    # Try to find local checkpoint
                    import glob
                    pattern = str(self.cache_dir / "models--McGill-NLP--ssa-comet-qe" / "snapshots" / "*" / "checkpoints" / "model.ckpt")
                    checkpoints = glob.glob(pattern)

                    if checkpoints:
                        model_path = checkpoints[0]
                        logger.info(f"Loading from local checkpoint: {model_path}")
                    else:
                        raise RuntimeError(f"Model download failed and no local checkpoint found at {pattern}")
                else:
                    raise

            # Load checkpoint with offline mode enabled
            self.model = load_from_checkpoint(model_path)

            # Try to move to device, fall back to CPU if CUDA is unavailable
            try:
                self.model.to(self.device)
            except RuntimeError as e:
                if "CUDA" in str(e) and self.device.type == "cuda":
                    logger.warning(f"CUDA device unavailable: {e}. Falling back to CPU.")
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                else:
                    raise

            self.model.eval()

            self._model_loaded = True
            logger.info(f"COMET model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load COMET model: {e}")
            raise RuntimeError(f"Could not load COMET model '{self.model_name}': {e}")
    
    def evaluate(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> Dict:
        """
        Evaluate translations using COMET.
        
        Args:
            sources: List of source sentences
            hypotheses: List of hypothesis translations
            references: List of reference translations
            
        Returns:
            Dictionary with scores and metadata
        """
        # Load model if needed
        self._load_model()
        
        if len(sources) != len(hypotheses) or len(sources) != len(references):
            raise ValueError("Sources, hypotheses, and references must have the same length")
        
        try:
            # Prepare data in COMET format
            data = [
                {
                    "src": src,
                    "mt": hyp,
                    "ref": ref,
                }
                for src, hyp, ref in zip(sources, hypotheses, references)
            ]
            
            # Compute scores
            logger.info(f"Computing COMET scores for {len(data)} samples...")
            
            with torch.no_grad():
                model_output = self.model.predict(
                    data,
                    batch_size=self.batch_size,
                    gpus=1 if self.device.type == "cuda" else 0,
                )
            
            return {
                'corpus_score': model_output.system_score,
                'sentence_scores': model_output.scores,
                'signature': f"model:{self.model_name}|device:{self.device}",
                'num_samples': len(data),
            }
        
        except Exception as e:
            logger.error(f"Error during COMET evaluation: {e}")
            return {
                'corpus_score': 0.0,
                'sentence_scores': [0.0] * len(sources),
                'signature': 'error',
                'error': str(e),
            }
    
    def evaluate_batch(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        chunk_size: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate translations in chunks (useful for large datasets).
        
        Args:
            sources: List of source sentences
            hypotheses: List of hypothesis translations
            references: List of reference translations
            chunk_size: Process in chunks of this size (default: batch_size * 10)
            
        Returns:
            Dictionary with scores and metadata
        """
        if chunk_size is None:
            chunk_size = self.batch_size * 10
        
        all_sentence_scores = []
        total_samples = len(sources)
        
        for i in range(0, total_samples, chunk_size):
            chunk_src = sources[i:i + chunk_size]
            chunk_hyp = hypotheses[i:i + chunk_size]
            chunk_ref = references[i:i + chunk_size]
            
            logger.info(f"Processing chunk {i // chunk_size + 1}/{(total_samples + chunk_size - 1) // chunk_size}")
            
            result = self.evaluate(chunk_src, chunk_hyp, chunk_ref)
            all_sentence_scores.extend(result['sentence_scores'])
        
        # Compute overall corpus score (mean of all sentence scores)
        corpus_score = sum(all_sentence_scores) / len(all_sentence_scores) if all_sentence_scores else 0.0
        
        return {
            'corpus_score': corpus_score,
            'sentence_scores': all_sentence_scores,
            'signature': f"model:{self.model_name}|device:{self.device}",
            'num_samples': total_samples,
        }
