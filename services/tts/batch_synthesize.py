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
        descriptor: str = None,
        device: str = "auto",
        sample_rate: int = 16000,
        pipeline_id: int = None,
    ):
        """
        Initialize the batch synthesizer.

        Args:
            model_path: Path to the fine-tuned XTTS model checkpoint
            reference_audio: Path to reference speaker audio for voice cloning
            descriptor: Descriptor for output directory naming (e.g., "nllb_tgt", "src", "custom_lang")
            device: Device to use ("cuda", "cpu", or "auto")
            sample_rate: Output sample rate (16000 for evaluation compatibility)
            pipeline_id: Optional pipeline ID (1-8) for filename generation
        """
        self.model_path = Path(model_path)
        self.reference_audio = Path(reference_audio)
        self.descriptor = descriptor
        self.target_sample_rate = sample_rate
        self.pipeline_id = pipeline_id

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                try:
                    # Test if CUDA is actually usable
                    _ = torch.tensor([1.0]).cuda()
                    self.device = "cuda"
                    logger.info("CUDA is available and working")
                except Exception as e:
                    logger.warning(f"CUDA available but not usable: {e}")
                    logger.info("Falling back to CPU")
                    self.device = "cpu"
            else:
                self.device = "cpu"
                logger.info("CUDA not available, using CPU")
        else:
            self.device = device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"

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

        # Move model to device with error handling
        try:
            self.model = self.model.to(self.device)
            logger.info(f"Model successfully moved to {self.device}")
        except Exception as e:
            if self.device == "cuda":
                logger.error(f"Failed to move model to CUDA: {e}")
                logger.info("Falling back to CPU")
                self.device = "cpu"
                self.model = self.model.to(self.device)
            else:
                raise

        self.model.eval()

        logger.info("Model loaded successfully")

        # Compute speaker latents from reference audio (once)
        logger.info(f"Computing speaker latents from {self.reference_audio}")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[str(self.reference_audio)]
        )

        logger.info("Speaker latents computed")

        # Debug: Check available languages in model
        if hasattr(self.model, 'language_manager') and hasattr(self.model.language_manager, 'language_id_mapping'):
            logger.info(f"Model language ID mapping: {self.model.language_manager.language_id_mapping}")
        elif hasattr(self.model, 'args') and hasattr(self.model.args, 'languages'):
            logger.info(f"Model languages from args: {self.model.args.languages}")

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
            if not language:
                logger.error(f"Language parameter is None or empty")
                return None

            # Debug: Log what we're about to synthesize
            logger.debug(f"Attempting synthesis with language='{language}', text_length={len(text)}")

            # Generate audio (24kHz by default)
            with torch.no_grad():
                out = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=self.gpt_cond_latent,
                    speaker_embedding=self.speaker_embedding,
                )

            logger.debug(f"Synthesis successful for language '{language}'")

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
            import traceback
            error_msg = str(e)
            if "'NoneType' object has no attribute 'encode'" in error_msg:
                logger.error(f"Synthesis failed - Language lookup error!")
                logger.error(f"Language code '{language}' may not be properly registered in the model")
                logger.error(f"Full error: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                logger.error(f"Text: {text[:50]}...")
            else:
                logger.error(f"Synthesis failed for text: {text[:50]}... Error: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
            return None

    def process_csv(
        self,
        csv_path: Path,
        output_base_dir: Path,
        language: str,
        text_column: str = "text",
        max_samples: Optional[int] = None,
        model_name: str = "xtts_v2",
    ) -> Dict:
        """
        Process samples from a CSV file.

        Args:
            csv_path: Path to input CSV file
            output_base_dir: Base directory for data (e.g., /path/to/data/languages)
            language: Language code for TTS (e.g., 'efi', 'ibo', 'swa', 'xho')
            text_column: Name of the column containing text to synthesize
            max_samples: Maximum number of samples to process (None = all)
            model_name: Model name (auto-extracted from model_path)

        Returns:
            Dictionary with processing statistics
        """
        # Get XTTS language code
        xtts_lang = ISO_TO_XTTS.get(language, language)

        if not xtts_lang:
            logger.error(f"Invalid language code: {language}")
            logger.error(f"Available language codes: {list(ISO_TO_XTTS.keys())}")
            return {"csv_path": str(csv_path), "status": "error", "samples": 0}

        # Extract model checkpoint name
        model_checkpoint = self.model_path.name

        # Construct output directory: /path/to/data/languages/{lang}/predicted_{descriptor}_{checkpoint}/
        output_base_dir = Path(output_base_dir)

        # Map to full language name if needed
        full_lang_name = {
            'efi': 'efik',
            'ibo': 'igbo',
            'swa': 'swahili',
            'swh': 'swahili',
            'xho': 'xhosa'
        }.get(language, language)

        if self.descriptor:
            audio_output_dir = output_base_dir / full_lang_name / f"predicted_{self.descriptor}_{model_checkpoint}"
            metadata_csv_path = output_base_dir / full_lang_name / f"predicted_{self.descriptor}_{model_checkpoint}.csv"
        else:
            # Fallback if no descriptor provided
            audio_output_dir = output_base_dir / full_lang_name / f"predicted_{model_checkpoint}"
            metadata_csv_path = output_base_dir / full_lang_name / f"predicted_{model_checkpoint}.csv"

        audio_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {language.upper()} -> {xtts_lang}")
        logger.info(f"{'='*60}")
        logger.info(f"Input CSV: {csv_path}")
        logger.info(f"Text column: {text_column}")
        logger.info(f"Audio output: {audio_output_dir}")
        logger.info(f"Metadata CSV: {metadata_csv_path}")
        logger.info(f"Max samples: {max_samples if max_samples else 'all'}")

        # Check if input exists
        if not csv_path.exists():
            logger.error(f"Input CSV not found: {csv_path}")
            return {"csv_path": str(csv_path), "status": "error", "samples": 0}

        # Load data
        logger.info("Loading data...")
        df = pd.read_csv(csv_path, sep="|")
        total_samples_in_file = len(df)
        logger.info(f"Loaded {total_samples_in_file} samples from CSV")

        # Verify text column exists
        if text_column not in df.columns:
            logger.error(f"Text column '{text_column}' not found. Available columns: {df.columns.tolist()}")
            return {"csv_path": str(csv_path), "status": "error", "samples": 0}

        # Limit samples if specified
        if max_samples and max_samples < total_samples_in_file:
            df = df.head(max_samples)
            logger.info(f"Limited to {max_samples} samples")

        total_samples = len(df)

        # Process each sample
        successful = 0
        failed = 0
        errors = []
        metadata_rows = []  # Track metadata for CSV generation

        logger.info(f"Synthesizing audio for {total_samples} samples...")

        # Determine if we have segment_id and user_id for filename generation
        has_segment_id = 'segment_id' in df.columns
        has_user_id = 'user_id' in df.columns

        with tqdm(total=total_samples, desc=f"{language}", unit="sample") as pbar:
            for idx, row in df.iterrows():
                text = row[text_column]

                # Extract segment_id and user_id if available
                segment_id = row.get('segment_id', idx)
                user_id = row.get('user_id', '')

                # Skip empty text
                if pd.isna(text) or not str(text).strip():
                    logger.warning(f"Skipping empty text at index {idx}")
                    failed += 1
                    # Track in metadata even if failed
                    metadata_rows.append({
                        'segment_id': segment_id,
                        'user_id': user_id,
                        'text': '',
                        'audio_filename': '',
                        'language': language,
                        'success': False,
                        'error': 'Empty text'
                    })
                    pbar.update(1)
                    continue

                # Generate output filename
                if has_segment_id and has_user_id:
                    if self.pipeline_id:
                        output_filename = f"Segment={segment_id}_User={user_id}_Language={language}_Pipeline={self.pipeline_id}_pred.wav"
                    else:
                        output_filename = f"Segment={segment_id}_User={user_id}_Language={language}_pred.wav"
                elif has_segment_id:
                    if self.pipeline_id:
                        output_filename = f"Segment={segment_id}_Language={language}_Pipeline={self.pipeline_id}_pred.wav"
                    else:
                        output_filename = f"Segment={segment_id}_Language={language}_pred.wav"
                else:
                    if self.pipeline_id:
                        output_filename = f"sample_{idx:05d}_Language={language}_Pipeline={self.pipeline_id}_pred.wav"
                    else:
                        output_filename = f"sample_{idx:05d}_Language={language}_pred.wav"

                output_path = audio_output_dir / output_filename

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
                        # Track successful synthesis in metadata
                        metadata_rows.append({
                            'segment_id': segment_id,
                            'user_id': user_id,
                            'text': str(text),
                            'audio_filename': output_filename,
                            'language': language,
                            'success': True,
                            'error': ''
                        })
                    except Exception as e:
                        logger.error(f"Failed to save audio for {output_filename}: {e}")
                        failed += 1
                        errors.append({
                            'index': idx,
                            'filename': output_filename,
                            'error': str(e)
                        })
                        # Track failure in metadata
                        metadata_rows.append({
                            'segment_id': segment_id,
                            'user_id': user_id,
                            'text': str(text),
                            'audio_filename': output_filename,
                            'language': language,
                            'success': False,
                            'error': f'Save failed: {str(e)}'
                        })
                else:
                    failed += 1
                    errors.append({
                        'index': idx,
                        'filename': output_filename,
                        'error': 'Synthesis failed'
                    })
                    # Track failure in metadata
                    metadata_rows.append({
                        'segment_id': segment_id,
                        'user_id': user_id,
                        'text': str(text),
                        'audio_filename': output_filename,
                        'language': language,
                        'success': False,
                        'error': 'Synthesis failed'
                    })

                # Clear GPU cache periodically
                if self.device == "cuda" and idx % 100 == 0:
                    torch.cuda.empty_cache()

                pbar.update(1)

        logger.info(f"Completed processing:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Success rate: {successful/total_samples*100:.2f}%")

        # Write metadata CSV
        if metadata_rows:
            logger.info(f"Writing metadata CSV to: {metadata_csv_path}")
            metadata_df = pd.DataFrame(metadata_rows)
            metadata_df.to_csv(metadata_csv_path, sep='|', index=False)
            logger.info(f"Metadata CSV saved with {len(metadata_rows)} rows")

        return {
            'csv_path': str(csv_path),
            'language': language,
            'status': 'success',
            'total_samples': total_samples,
            'successful': successful,
            'failed': failed,
            'errors': len(errors),
            'metadata_csv': str(metadata_csv_path),
            'audio_output_dir': str(audio_output_dir),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Batch TTS synthesis - flexible CSV processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single CSV file
  python batch_synthesize.py --csv-file data.csv --output-dir ./audio_output --language swa

  # Process with custom text column and limit
  python batch_synthesize.py --csv-file data.csv --output-dir ./output \\
      --language ibo --text-column translated_text --max-samples 100

  # Use custom model
  python batch_synthesize.py --csv-file data.csv --output-dir ./output \\
      --language efi --model-path ./my_model --reference-audio ./ref.wav
        """
    )

    # Required arguments
    parser.add_argument(
        '--csv-file',
        type=str,
        required=True,
        help='Path to input CSV file (pipe-delimited)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save synthesized audio files'
    )
    parser.add_argument(
        '--language',
        type=str,
        required=True,
        help='Language code for TTS (e.g., efi, ibo, swa, xho, swh)'
    )

    # Optional arguments
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Name of the column containing text to synthesize (default: "text")'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (default: all)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='/home/vacl2/multimodal_translation/services/tts/checkpoints/MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632',
        help='Path to fine-tuned XTTS model directory'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name for output folder suffix (auto-extracted from model-path if not provided)'
    )
    parser.add_argument(
        '--reference-audio',
        type=str,
        default='/home/vacl2/multimodal_translation/services/tts/reference_audio/female_en.wav',
        help='Path to reference speaker audio for voice cloning'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Output sample rate in Hz (default: 16000)'
    )
    parser.add_argument(
        '--pipeline-id',
        type=int,
        default=None,
        help='Pipeline ID (1-8) for multi-pipeline evaluation (optional)'
    )
    parser.add_argument(
        '--descriptor',
        type=str,
        default=None,
        help='Descriptor for output directory naming (e.g., "nllb_tgt", "src", "custom_lang")'
    )

    args = parser.parse_args()

    # Validate paths
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    # Auto-extract model name from path if not provided
    if not args.model_name:
        args.model_name = Path(args.model_path).name

    logger.info(f"{'='*60}")
    logger.info("BATCH TTS SYNTHESIS")
    logger.info(f"{'='*60}")
    logger.info(f"CSV file: {csv_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Text column: {args.text_column}")
    logger.info(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Reference audio: {args.reference_audio}")
    logger.info(f"Device: {args.device}")

    # Warn about CPU performance
    if args.device == "cpu":
        logger.warning("Running on CPU - synthesis will be significantly slower than GPU")
        logger.warning("Consider using --device cuda if GPU is available")

    try:
        # Initialize synthesizer
        logger.info("\nInitializing TTS model...")
        synthesizer = TTSBatchSynthesizer(
            model_path=args.model_path,
            reference_audio=args.reference_audio,
            descriptor=args.descriptor,
            device=args.device,
            sample_rate=args.sample_rate,
            pipeline_id=args.pipeline_id,
        )

        # Process CSV
        result = synthesizer.process_csv(
            csv_path=csv_path,
            output_base_dir=Path(args.output_dir),
            language=args.language,
            text_column=args.text_column,
            max_samples=args.max_samples,
            model_name=args.model_name,
        )

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("SYNTHESIS COMPLETE")
        logger.info(f"{'='*60}")

        if result['status'] == 'success':
            logger.info(f"Total samples: {result['total_samples']}")
            logger.info(f"Successful: {result['successful']}")
            logger.info(f"Failed: {result['failed']}")
            logger.info(f"Success rate: {result['successful']/result['total_samples']*100:.2f}%")
            logger.info(f"\nOutput directory: {args.output_dir}")
        else:
            logger.error("Processing failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Batch synthesis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
