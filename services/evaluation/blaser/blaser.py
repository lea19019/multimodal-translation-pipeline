"""
BLASER 2.0 evaluation module.

Provides speech-to-speech translation quality assessment using BLASER 2.0
from Meta's SONAR framework.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class BlaserEvaluator:
    """Evaluate speech-to-speech translation using BLASER 2.0."""
    
    def __init__(
        self,
        model_name: str = "blaser_2_0_ref",
        device: Optional[torch.device] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize BLASER evaluator.
        
        Args:
            model_name: BLASER model name ('blaser_2_0_ref' or 'blaser_2_0_qe')
            device: Torch device (auto-detect if None)
            cache_dir: Directory to cache models (default: ./blaser/models/)
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Set cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "models"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Models loaded lazily
        self.blaser_model = None
        self.text_embedder = None
        self.speech_embedder = None
        self._models_loaded = False
    
    def _load_models(self):
        """Load BLASER and SONAR models."""
        if self._models_loaded:
            return
        
        try:
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
            from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
            from sonar.models.blaser.loader import load_blaser_model
            
            logger.info(f"Loading BLASER model: {self.model_name}")
            
            # Load BLASER model
            self.blaser_model = load_blaser_model(self.model_name).eval()
            self.blaser_model.to(self.device)
            
            # Load SONAR text embedder
            logger.info("Loading SONAR text embedder...")
            self.text_embedder = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device=self.device,
            )
            
            # Note: Speech embedder will be loaded per-language when needed
            self.speech_embedder = None
            
            self._models_loaded = True
            logger.info(f"BLASER models loaded successfully on {self.device}")
        
        except Exception as e:
            logger.error(f"Failed to load BLASER models: {e}")
            raise RuntimeError(f"Could not load BLASER model '{self.model_name}': {e}")
    
    def _load_speech_embedder(self, language: str):
        """
        Load speech embedder for specific language.
        
        Args:
            language: Language code (e.g., 'eng', 'spa')
        """
        try:
            from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
            
            # Map common language codes to SONAR encoder names
            # Note: This mapping may need to be extended
            lang_to_encoder = {
                'en': 'sonar_speech_encoder_eng',
                'eng': 'sonar_speech_encoder_eng',
                'es': 'sonar_speech_encoder_spa',
                'spa': 'sonar_speech_encoder_spa',
                # Add more languages as needed
            }
            
            encoder_name = lang_to_encoder.get(language.lower())
            if not encoder_name:
                logger.warning(f"No speech encoder found for language '{language}', using English")
                encoder_name = 'sonar_speech_encoder_eng'
            
            logger.info(f"Loading speech embedder for language: {language}")
            self.speech_embedder = SpeechToEmbeddingModelPipeline(
                encoder=encoder_name,
                device=self.device,
            )
        
        except Exception as e:
            logger.error(f"Failed to load speech embedder for {language}: {e}")
            raise
    
    def evaluate(
        self,
        source_audio_paths: List[Union[str, Path]],
        target_audio_paths: List[Union[str, Path]],
        reference_texts: List[str],
        source_texts: List[str],
        source_lang: str = 'eng_Latn',
        target_lang: str = 'spa_Latn',
    ) -> Dict:
        """
        Evaluate speech-to-speech translations using BLASER 2.0.
        
        Args:
            source_audio_paths: Paths to source audio files
            target_audio_paths: Paths to target audio files
            reference_texts: Reference translations (text)
            source_texts: Source texts
            source_lang: Source language code (e.g., 'eng_Latn')
            target_lang: Target language code (e.g., 'spa_Latn')
            
        Returns:
            Dictionary with BLASER scores
        """
        # Load models if needed
        self._load_models()
        
        if len(source_audio_paths) != len(target_audio_paths):
            raise ValueError("Source and target audio lists must have same length")
        
        try:
            # Convert paths to strings
            source_audio_paths = [str(p) for p in source_audio_paths]
            target_audio_paths = [str(p) for p in target_audio_paths]
            
            # Load speech embedder for source language
            source_lang_code = source_lang.split('_')[0] if '_' in source_lang else source_lang
            self._load_speech_embedder(source_lang_code)
            
            logger.info("Computing SONAR embeddings for source audio...")
            src_embs = self.speech_embedder.predict(source_audio_paths)
            
            # For target audio, reload embedder if needed
            target_lang_code = target_lang.split('_')[0] if '_' in target_lang else target_lang
            if target_lang_code != source_lang_code:
                self._load_speech_embedder(target_lang_code)
            
            logger.info("Computing SONAR embeddings for target audio...")
            mt_embs = self.speech_embedder.predict(target_audio_paths)
            
            # Compute text embeddings for references
            logger.info("Computing SONAR embeddings for reference texts...")
            ref_embs = self.text_embedder.predict(reference_texts, source_lang=target_lang)
            
            # Compute text embeddings for sources (for ref-based BLASER)
            src_text_embs = self.text_embedder.predict(source_texts, source_lang=source_lang)
            
            # Compute BLASER scores
            logger.info(f"Computing BLASER scores for {len(source_audio_paths)} samples...")
            
            scores = []
            with torch.inference_mode():
                for i in range(len(source_audio_paths)):
                    if self.model_name == "blaser_2_0_ref":
                        # Reference-based BLASER
                        score = self.blaser_model(
                            src=src_embs[i:i+1].to(self.device),
                            ref=ref_embs[i:i+1].to(self.device),
                            mt=mt_embs[i:i+1].to(self.device),
                        ).item()
                    else:
                        # QE-based BLASER (no reference)
                        score = self.blaser_model(
                            src=src_embs[i:i+1].to(self.device),
                            mt=mt_embs[i:i+1].to(self.device),
                        ).item()
                    scores.append(score)
            
            mean_score = sum(scores) / len(scores) if scores else 0.0
            
            return {
                'corpus_score': mean_score,
                'sentence_scores': scores,
                'signature': f"model:{self.model_name}|device:{self.device}",
                'num_samples': len(scores),
            }
        
        except Exception as e:
            logger.error(f"Error during BLASER evaluation: {e}")
            return {
                'corpus_score': 0.0,
                'sentence_scores': [0.0] * len(source_audio_paths),
                'signature': 'error',
                'error': str(e),
            }


def evaluate_blaser(
    source_audio_paths: List[Union[str, Path]],
    target_audio_paths: List[Union[str, Path]],
    reference_texts: List[str],
    source_texts: List[str],
    source_lang: str = 'eng_Latn',
    target_lang: str = 'spa_Latn',
    model_name: str = "blaser_2_0_ref",
) -> Dict:
    """
    Convenience function to evaluate with BLASER 2.0.
    
    Args:
        source_audio_paths: Paths to source audio files
        target_audio_paths: Paths to target audio files
        reference_texts: Reference translations
        source_texts: Source texts
        source_lang: Source language
        target_lang: Target language
        model_name: BLASER model name
        
    Returns:
        Dictionary with BLASER scores
    """
    evaluator = BlaserEvaluator(model_name=model_name)
    return evaluator.evaluate(
        source_audio_paths,
        target_audio_paths,
        reference_texts,
        source_texts,
        source_lang,
        target_lang,
    )