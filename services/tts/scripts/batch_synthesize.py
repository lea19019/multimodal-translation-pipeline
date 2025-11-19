#!/usr/bin/env python3
"""
Batch TTS Synthesis Script for Multimodal Translation Evaluation

This script performs GPU-accelerated batch audio synthesis using the fine-tuned XTTS model
to generate predicted target language audio for evaluation purposes.

Usage:
    python batch_synthesize.py --languages efik igbo swahili xhosa --input-source nmt_predictions

Output:
    Creates predicted_tgt_audio/ folder in each language directory with WAV files:
    Segment={segment_id}_User={user_id}_Language={iso}_pred.wav
    Format: 16kHz, 16-bit PCM, mono
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import torch
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Language code mapping
LANGUAGE_NAME_MAPPING = {
    'efik': 'efi',
    'igbo': 'ibo',
    'swahili': 'swa',
    'xhosa': 'xho',
}

# XTTS language codes (different from NMT)
ISO_TO_XTTS = {
    'efi': 'efi',
    'ibo': 'ibo',
    'swa': 'swa',  # Swahili (CSV uses 'swa')
    'swh': 'swa',  # Alternative Swahili code
    'xho': 'xho',  # Xhosa
}


class TTSBatchSynthesizer:
    """Handles batch TTS synthesis for multiple languages."""

    def __init__(
        self,
        model_path: str,
        reference_audio: str,
        device: str = "auto",
        sample_rate: int = 16000,
    ):
        """
        Initialize the batch synthesizer.

        Args:
            model_path: Path to the fine-tuned XTTS model checkpoint
            reference_audio: Path to reference speaker audio for voice cloning
            device: Device to use ("cuda", "cpu", or "auto")
            sample_rate: Output sample rate (16000 for evaluation compatibility)
        """
        self.model_path = Path(model_path)
        self.reference_audio = Path(reference_audio)
        self.target_sample_rate = sample_rate

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Verify reference audio exists
        if not self.reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {self.reference_audio}")

        # Load model
        logger.info(f"Loading XTTS model from {self.model_path}")
        config_path = self.model_path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        config = XttsConfig()
        config.load_json(str(config_path))

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_dir=str(self.model_path),
            use_deepspeed=False
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

        # Compute speaker latents from reference audio (once)
        logger.info(f"Computing speaker latents from {self.reference_audio}")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[str(self.reference_audio)]
        )

        logger.info("Speaker latents computed")

    def synthesize(
        self,
        text: str,
        language: str,
    ) -> Optional[np.ndarray]:
        """
        Synthesize audio from text.

        Args:
            text: Text to synthesize
            language: Language code (XTTS format, e.g., 'efi', 'swa')

        Returns:
            Audio waveform as numpy array at target sample rate, or None if failed
        """
        try:
            # Generate audio (24kHz by default)
            with torch.no_grad():
                out = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=self.gpt_cond_latent,
                    speaker_embedding=self.speaker_embedding,
                )

            # Get waveform (returns dict with 'wav' key)
            wav = out['wav']

            # Convert to numpy if tensor
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            # XTTS outputs at 24kHz, resample to target (16kHz for evaluation)
            if self.target_sample_rate != 24000:
                wav = librosa.resample(
                    wav,
                    orig_sr=24000,
                    target_sr=self.target_sample_rate
                )

            # Normalize to [-1.0, 1.0]
            wav = wav / np.max(np.abs(wav)) if np.max(np.abs(wav)) > 0 else wav

            return wav

        except Exception as e:
            logger.error(f"Synthesis failed for text: {text[:50]}... Error: {e}")
            return None

    def process_language(
        self,
        language: str,
        data_dir: Path,
        input_source: str = "nmt_predictions",
    ) -> Dict:
        """
        Process all test samples for a single language.

        Args:
            language: Language name (efik, igbo, swahili, xhosa)
            data_dir: Base data directory containing language folders
            input_source: "ground_truth" or "nmt_predictions"

        Returns:
            Dictionary with processing statistics
        """
        # Map language name to ISO code
        iso_code = LANGUAGE_NAME_MAPPING.get(language.lower())
        if not iso_code:
            raise ValueError(f"Unknown language: {language}")

        # Get XTTS language code
        xtts_lang = ISO_TO_XTTS[iso_code]

        # Paths
        lang_dir = data_dir / language

        if input_source == "nmt_predictions":
            input_csv = lang_dir / "nmt_predictions.csv"
            text_column = "predicted_tgt_text"
        else:  # ground_truth
            input_csv = lang_dir / "mapped_metadata_test.csv"
            text_column = "tgt_text"

        output_dir = lang_dir / "predicted_tgt_audio"
        output_dir.mkdir(exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {language.upper()} ({iso_code} -> {xtts_lang})")
        logger.info(f"{'='*60}")
        logger.info(f"Input: {input_csv}")
        logger.info(f"Text column: {text_column}")
        logger.info(f"Output: {output_dir}")

        # Check if input exists
        if not input_csv.exists():
            logger.error(f"Input CSV not found: {input_csv}")
            return {"language": language, "status": "error", "samples": 0}

        # Load data
        logger.info("Loading data...")
        df = pd.read_csv(input_csv, sep="|")
        total_samples = len(df)
        logger.info(f"Loaded {total_samples} samples")

        # Verify required columns
        required_cols = ['segment_id', 'user_id', text_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return {"language": language, "status": "error", "samples": 0}

        # Process each sample
        successful = 0
        failed = 0
        errors = []

        logger.info(f"Synthesizing audio for {total_samples} samples...")

        with tqdm(total=total_samples, desc=f"{language}", unit="sample") as pbar:
            for idx, row in df.iterrows():
                segment_id = row['segment_id']
                user_id = row['user_id']
                text = row[text_column]

                # Skip empty text
                if pd.isna(text) or not str(text).strip():
                    logger.warning(f"Skipping empty text for segment={segment_id}, user={user_id}")
                    failed += 1
                    pbar.update(1)
                    continue

                # Output filename
                output_filename = f"Segment={segment_id}_User={user_id}_Language={iso_code}_pred.wav"
                output_path = output_dir / output_filename

                # Synthesize audio
                wav = self.synthesize(str(text), xtts_lang)

                if wav is not None:
                    try:
                        # Save as 16-bit PCM WAV
                        sf.write(
                            output_path,
                            wav,
                            self.target_sample_rate,
                            subtype='PCM_16'
                        )
                        successful += 1
                    except Exception as e:
                        logger.error(f"Failed to save audio for {output_filename}: {e}")
                        failed += 1
                        errors.append({
                            'segment_id': segment_id,
                            'user_id': user_id,
                            'error': str(e)
                        })
                else:
                    failed += 1
                    errors.append({
                        'segment_id': segment_id,
                        'user_id': user_id,
                        'error': 'Synthesis failed'
                    })

                # Clear GPU cache periodically
                if self.device == "cuda" and idx % 100 == 0:
                    torch.cuda.empty_cache()

                pbar.update(1)

        logger.info(f"Completed {language}:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Success rate: {successful/total_samples*100:.2f}%")

        return {
            'language': language,
            'status': 'success',
            'total_samples': total_samples,
            'successful': successful,
            'failed': failed,
            'errors': len(errors),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Batch TTS synthesis for multimodal translation evaluation"
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['efik', 'igbo', 'swahili', 'xhosa'],
        choices=['efik', 'igbo', 'swahili', 'xhosa'],
        help='Languages to process (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/home/vacl2/multimodal_translation/services/data/languages',
        help='Base directory containing language folders'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='/home/vacl2/multimodal_translation/services/tts/checkpoints/MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632',
        help='Path to fine-tuned XTTS model'
    )
    parser.add_argument(
        '--reference-audio',
        type=str,
        default='/home/vacl2/multimodal_translation/services/tts/reference_audio/female_en.wav',
        help='Path to reference speaker audio'
    )
    parser.add_argument(
        '--input-source',
        type=str,
        default='nmt_predictions',
        choices=['ground_truth', 'nmt_predictions'],
        help='Text source: ground_truth (tgt_text from CSV) or nmt_predictions (predicted_tgt_text)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Output sample rate in Hz (default: 16000 for evaluation compatibility)'
    )

    args = parser.parse_args()

    try:
        # Initialize synthesizer
        synthesizer = TTSBatchSynthesizer(
            model_path=args.model_path,
            reference_audio=args.reference_audio,
            device=args.device,
            sample_rate=args.sample_rate,
        )

        # Process each language
        data_dir = Path(args.data_dir)
        results = []

        for language in args.languages:
            result = synthesizer.process_language(
                language,
                data_dir,
                input_source=args.input_source,
            )
            results.append(result)

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("BATCH SYNTHESIS COMPLETE")
        logger.info(f"{'='*60}")

        total_samples = sum(r['total_samples'] for r in results if r['status'] == 'success')
        total_successful = sum(r['successful'] for r in results if r['status'] == 'success')

        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Successful synthesis: {total_successful}")
        logger.info(f"Success rate: {total_successful/total_samples*100:.2f}%")

        logger.info("\nPer-language results:")
        for result in results:
            if result['status'] == 'success':
                logger.info(f"  {result['language']:10s}: {result['successful']:5d}/{result['total_samples']:5d} ({result['successful']/result['total_samples']*100:.1f}%)")
            else:
                logger.info(f"  {result['language']:10s}: ERROR")

    except Exception as e:
        logger.error(f"Batch synthesis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
