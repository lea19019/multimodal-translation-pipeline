#!/usr/bin/env python3
"""
Download BLASER 2.0 and SONAR models for speech-to-speech evaluation.

This script downloads:
- BLASER 2.0 QE model (reference-free quality estimation)
- BLASER 2.0 REF model (reference-based evaluation)
- SONAR text encoders
- SONAR speech encoders for supported languages

Models are cached in ../models/ directory.
"""

import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_blaser_models(model_dir: Path, models: list[str]):
    """Download BLASER models."""
    try:
        from sonar.models.blaser.loader import load_blaser_model

        for model_name in models:
            logger.info(f"Downloading BLASER model: {model_name}")
            try:
                model = load_blaser_model(model_name)
                logger.info(f"✓ Successfully loaded {model_name}")
                del model  # Free memory
            except Exception as e:
                logger.error(f"✗ Failed to load {model_name}: {e}")
                raise

    except ImportError as e:
        logger.error(f"Failed to import SONAR: {e}")
        logger.error("Make sure fairseq2 and sonar-space are installed")
        raise


def download_sonar_text_encoder(model_dir: Path):
    """Download SONAR text encoder."""
    try:
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

        logger.info("Downloading SONAR text encoder...")
        encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
        )
        logger.info("✓ Successfully loaded SONAR text encoder")

        # Test encoding
        test_emb = encoder.predict(["Hello world"], source_lang="eng_Latn")
        logger.info(f"✓ Text encoder test successful (embedding shape: {test_emb.shape})")

        del encoder

    except Exception as e:
        logger.error(f"✗ Failed to load SONAR text encoder: {e}")
        raise


def download_sonar_speech_encoders(model_dir: Path, languages: list[str]):
    """Download SONAR speech encoders for specified languages."""
    try:
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

        # Language code to encoder mapping
        lang_to_encoder = {
            'eng': 'sonar_speech_encoder_eng',
            'spa': 'sonar_speech_encoder_spa',
            'fra': 'sonar_speech_encoder_fra',
            'deu': 'sonar_speech_encoder_deu',
            'ita': 'sonar_speech_encoder_ita',
            'por': 'sonar_speech_encoder_por',
            'rus': 'sonar_speech_encoder_rus',
            'cmn': 'sonar_speech_encoder_cmn',
            'jpn': 'sonar_speech_encoder_jpn',
            'kor': 'sonar_speech_encoder_kor',
            'arb': 'sonar_speech_encoder_arb',
            'hin': 'sonar_speech_encoder_hin',
            'tel': 'sonar_speech_encoder_tel',
            'urd': 'sonar_speech_encoder_urd',
            'vie': 'sonar_speech_encoder_vie',
            'tha': 'sonar_speech_encoder_tha',
            'ind': 'sonar_speech_encoder_ind',
            'msa': 'sonar_speech_encoder_msa',
            'tgl': 'sonar_speech_encoder_tgl',
            'swh': 'sonar_speech_encoder_swh',
            'afr': 'sonar_speech_encoder_afr',
            'amh': 'sonar_speech_encoder_amh',
            'yor': 'sonar_speech_encoder_yor',
            'ibo': 'sonar_speech_encoder_ibo',
            'zul': 'sonar_speech_encoder_zul',
            # Add more as needed
        }

        for lang in languages:
            encoder_name = lang_to_encoder.get(lang)
            if not encoder_name:
                logger.warning(f"No speech encoder found for language: {lang}, skipping")
                continue

            logger.info(f"Downloading SONAR speech encoder for {lang}...")
            try:
                encoder = SpeechToEmbeddingModelPipeline(encoder=encoder_name)
                logger.info(f"✓ Successfully loaded speech encoder for {lang}")
                del encoder
            except Exception as e:
                logger.error(f"✗ Failed to load speech encoder for {lang}: {e}")
                # Don't raise - continue with other languages

    except Exception as e:
        logger.error(f"✗ Failed to import SONAR speech pipeline: {e}")
        raise


@click.command()
@click.option(
    '--blaser-models',
    default='blaser_2_0_qe,blaser_2_0_ref',
    help='Comma-separated list of BLASER models to download'
)
@click.option(
    '--languages',
    default='eng,spa',
    help='Comma-separated list of language codes for speech encoders'
)
@click.option(
    '--model-dir',
    type=click.Path(path_type=Path),
    default=None,
    help='Directory to cache models (default: ../models/)'
)
@click.option(
    '--skip-blaser',
    is_flag=True,
    help='Skip BLASER model download'
)
@click.option(
    '--skip-text',
    is_flag=True,
    help='Skip SONAR text encoder download'
)
@click.option(
    '--skip-speech',
    is_flag=True,
    help='Skip SONAR speech encoder download'
)
def main(
    blaser_models: str,
    languages: str,
    model_dir: Path | None,
    skip_blaser: bool,
    skip_text: bool,
    skip_speech: bool,
):
    """Download BLASER 2.0 and SONAR models."""

    # Load environment variables from .env file
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment from {env_file}")

    # Set model directory
    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "models"
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables for model caching
    os.environ['FAIRSEQ2_CACHE_DIR'] = str(model_dir)
    os.environ['HF_HOME'] = str(model_dir / 'huggingface')
    os.environ['TORCH_HOME'] = str(model_dir / 'torch')

    logger.info(f"Model cache directory: {model_dir}")
    logger.info(f"FAIRSEQ2_CACHE_DIR: {os.environ.get('FAIRSEQ2_CACHE_DIR')}")
    logger.info(f"HF_HOME: {os.environ.get('HF_HOME')}")
    logger.info("="*60)

    # Parse model lists
    blaser_model_list = [m.strip() for m in blaser_models.split(',')]
    language_list = [l.strip() for l in languages.split(',')]

    try:
        # Download BLASER models
        if not skip_blaser:
            logger.info("Downloading BLASER models...")
            download_blaser_models(model_dir, blaser_model_list)
            logger.info("")

        # Download SONAR text encoder
        if not skip_text:
            logger.info("Downloading SONAR text encoder...")
            download_sonar_text_encoder(model_dir)
            logger.info("")

        # Download SONAR speech encoders
        if not skip_speech:
            logger.info(f"Downloading SONAR speech encoders for languages: {language_list}")
            download_sonar_speech_encoders(model_dir, language_list)
            logger.info("")

        logger.info("="*60)
        logger.info("✓ All models downloaded successfully!")
        logger.info(f"Models cached in: {model_dir}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run verify_setup.py to test the installation")
        logger.info("  2. Use evaluate.py to run BLASER evaluation")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
