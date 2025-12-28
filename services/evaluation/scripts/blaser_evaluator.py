"""
BLASER 2.0 evaluation module.

Provides speech-to-speech translation quality assessment using BLASER 2.0
from Meta's SONAR framework.

This module uses a subprocess to call the BLASER evaluation script in a separate
environment (blaser/.venv) to avoid PyTorch version conflicts.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class BlaserEvaluator:
    """Evaluate speech-to-speech translation using BLASER 2.0 via subprocess."""

    def __init__(
        self,
        model_name: str = "blaser_2_0_qe",
        device: Optional[str] = None,
    ):
        """
        Initialize BLASER evaluator.

        Args:
            model_name: BLASER model name ('blaser_2_0_ref' or 'blaser_2_0_qe')
            device: Device string ('cpu' or 'cuda')
        """
        self.model_name = model_name

        # Set device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Locate BLASER environment
        self.blaser_dir = Path(__file__).parent.parent / "blaser"
        self.evaluate_script = self.blaser_dir / "evaluate.py"
        self.blaser_python = self.blaser_dir / ".venv" / "bin" / "python"

        # Check if BLASER environment exists
        if not self.blaser_dir.exists():
            raise RuntimeError(
                f"BLASER environment not found at {self.blaser_dir}. "
                "Please set up the BLASER environment first."
            )

        if not self.evaluate_script.exists():
            raise RuntimeError(
                f"BLASER evaluate script not found at {self.evaluate_script}"
            )

        if not self.blaser_python.exists():
            raise RuntimeError(
                f"BLASER Python not found at {self.blaser_python}. "
                "Please create the BLASER virtual environment first."
            )
    
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

        This method calls the BLASER evaluation script in a separate environment
        via subprocess to avoid PyTorch version conflicts.

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
        if len(source_audio_paths) != len(target_audio_paths):
            raise ValueError("Source and target audio lists must have same length")

        try:
            # Convert paths to strings
            source_audio_paths = [str(p) for p in source_audio_paths]
            target_audio_paths = [str(p) for p in target_audio_paths]

            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                output_file = tmp.name

            # Build command
            cmd = [
                str(self.blaser_python),
                str(self.evaluate_script),
                '--model-name', self.model_name,
                '--device', self.device,
                '--source-lang', source_lang,
                '--target-lang', target_lang,
                '--output', output_file,
            ]

            # Add source audio paths
            for path in source_audio_paths:
                cmd.extend(['--source-audio', path])

            # Add target audio paths
            for path in target_audio_paths:
                cmd.extend(['--target-audio', path])

            # Run BLASER evaluation in subprocess
            logger.info(f"Running BLASER evaluation for {len(source_audio_paths)} samples...")
            logger.debug(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            # Check for errors
            if result.returncode != 0:
                logger.error(f"BLASER evaluation failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"BLASER evaluation failed: {result.stderr}")

            # Read results from output file
            with open(output_file, 'r') as f:
                blaser_result = json.load(f)

            # Clean up temp file
            Path(output_file).unlink()

            # Add signature
            blaser_result['signature'] = f"model:{self.model_name}|device:{self.device}"

            logger.info(f"BLASER evaluation complete. Corpus score: {blaser_result['corpus_score']:.4f}")

            return blaser_result

        except subprocess.TimeoutExpired:
            logger.error("BLASER evaluation timed out")
            return {
                'corpus_score': 0.0,
                'sentence_scores': [0.0] * len(source_audio_paths),
                'signature': 'error',
                'error': 'Evaluation timed out',
            }
        except Exception as e:
            logger.error(f"Error during BLASER evaluation: {e}")
            return {
                'corpus_score': 0.0,
                'sentence_scores': [0.0] * len(source_audio_paths),
                'signature': 'error',
                'error': str(e),
            }
