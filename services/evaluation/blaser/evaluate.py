#!/usr/bin/env python3
"""
BLASER 2.0 Evaluation CLI

Command-line interface for speech-to-speech translation evaluation using BLASER 2.0.
This script is called by the main evaluation system via subprocess.

Usage:
    python evaluate.py --source-audio audio1.wav audio2.wav \\
                       --target-audio trans1.wav trans2.wav \\
                       --source-text "Hello" "World" \\
                       --reference-text "Hola" "Mundo" \\
                       --source-lang eng_Latn \\
                       --target-lang spa_Latn \\
                       --output results.json
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import click
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Language mapping for SONAR text encoder compatibility
# Some languages are not supported by SONAR text encoder, so we use proxies
SONAR_TEXT_LANG_MAP = {
    'efi_Latn': 'ibo_Latn',  # Efik uses Igbo as proxy (not supported by NLLB-200)
    'xho_Latn': 'xho_Latn',  # Xhosa is supported
    'ibo_Latn': 'ibo_Latn',  # Igbo is supported
    'swh_Latn': 'swh_Latn',  # Swahili is supported
    'eng_Latn': 'eng_Latn',  # English is supported
}


def map_language_for_text_encoder(lang: str) -> str:
    """Map unsupported languages to SONAR-compatible proxies for text encoding."""
    return SONAR_TEXT_LANG_MAP.get(lang, lang)


def load_blaser_model(model_name: str = "blaser_2_0_qe"):
    """Load BLASER model."""
    try:
        from sonar.models.blaser.loader import load_blaser_model as load_model

        logger.info(f"Loading BLASER model: {model_name}")
        model = load_model(model_name).eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load BLASER model: {e}")
        raise


def load_sonar_encoders(device: torch.device):
    """Load SONAR text and speech encoders."""
    try:
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

        logger.info("Loading SONAR encoders...")
        text_encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        )

        return text_encoder
    except Exception as e:
        logger.error(f"Failed to load SONAR encoders: {e}")
        raise


def load_speech_encoder(language: str, device: torch.device):
    """Load SONAR speech encoder for specific language."""
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
        }

        encoder_name = lang_to_encoder.get(language, 'sonar_speech_encoder_eng')
        logger.info(f"Loading speech encoder for language: {language} ({encoder_name})")

        speech_encoder = SpeechToEmbeddingModelPipeline(
            encoder=encoder_name,
            device=device,
        )
        return speech_encoder
    except Exception as e:
        logger.error(f"Failed to load speech encoder for {language}: {e}")
        raise


@click.command()
@click.option('--source-audio', multiple=True, required=True, help='Source audio file paths')
@click.option('--target-audio', multiple=True, required=True, help='Target audio file paths')
@click.option('--source-text', multiple=True, required=True, help='Source text')
@click.option('--reference-text', multiple=True, required=True, help='Reference text')
@click.option('--source-lang', default='eng_Latn', help='Source language code')
@click.option('--target-lang', default='spa_Latn', help='Target language code')
@click.option('--model-name', default='blaser_2_0_qe', help='BLASER model name')
@click.option('--output', type=click.Path(), required=True, help='Output JSON file')
@click.option('--device', default='cpu', help='Device (cpu/cuda)')
def main(
    source_audio: tuple,
    target_audio: tuple,
    source_text: tuple,
    reference_text: tuple,
    source_lang: str,
    target_lang: str,
    model_name: str,
    output: str,
    device: str,
):
    """Run BLASER 2.0 evaluation."""

    try:
        # Convert to lists
        source_audio_paths = list(source_audio)
        target_audio_paths = list(target_audio)
        source_texts = list(source_text)
        reference_texts = list(reference_text)

        # Validate inputs
        if len(source_audio_paths) != len(target_audio_paths):
            raise ValueError("Number of source and target audio files must match")
        if len(source_texts) != len(reference_texts):
            raise ValueError("Number of source and reference texts must match")
        if len(source_audio_paths) != len(source_texts):
            raise ValueError("Number of audio files must match number of texts")

        # Setup device
        device_obj = torch.device(device)
        logger.info(f"Using device: {device_obj}")

        # Load models
        blaser_model = load_blaser_model(model_name)

        # Try to move to device, fall back to CPU if CUDA is unavailable
        try:
            blaser_model.to(device_obj)
        except RuntimeError as e:
            if "CUDA" in str(e) and device_obj.type == "cuda":
                logger.warning(f"CUDA device unavailable: {e}. Falling back to CPU.")
                device_obj = torch.device("cpu")
                blaser_model.to(device_obj)
            else:
                raise

        text_encoder = load_sonar_encoders(device_obj)

        # Extract language codes (e.g., 'eng_Latn' -> 'eng')
        source_lang_code = source_lang.split('_')[0]
        target_lang_code = target_lang.split('_')[0]

        # Load speech encoders
        source_speech_encoder = load_speech_encoder(source_lang_code, device_obj)
        if source_lang_code != target_lang_code:
            target_speech_encoder = load_speech_encoder(target_lang_code, device_obj)
        else:
            target_speech_encoder = source_speech_encoder

        # Compute embeddings
        # NOTE: Audio files must be at 16kHz sample rate (SONAR requirement)
        # Use scripts/resample_audio_samples.py to preprocess audio files
        logger.info("Computing SONAR embeddings...")

        # Source audio embeddings
        logger.info(f"Processing {len(source_audio_paths)} source audio files...")
        src_embs = source_speech_encoder.predict(source_audio_paths)

        # Target audio embeddings
        logger.info(f"Processing {len(target_audio_paths)} target audio files...")
        mt_embs = target_speech_encoder.predict(target_audio_paths)

        # Reference text embeddings
        # Map languages to SONAR-compatible proxies for text encoding
        target_lang_mapped = map_language_for_text_encoder(target_lang)
        source_lang_mapped = map_language_for_text_encoder(source_lang)

        logger.info("Processing reference texts...")
        if target_lang != target_lang_mapped:
            logger.info(f"Using language proxy for text encoding: {target_lang} -> {target_lang_mapped}")
        ref_embs = text_encoder.predict(reference_texts, source_lang=target_lang_mapped)

        # Source text embeddings (for ref-based BLASER)
        if source_lang != source_lang_mapped:
            logger.info(f"Using language proxy for text encoding: {source_lang} -> {source_lang_mapped}")
        src_text_embs = text_encoder.predict(source_texts, source_lang=source_lang_mapped)

        # Compute BLASER scores
        logger.info(f"Computing BLASER scores for {len(source_audio_paths)} samples...")

        scores = []
        with torch.inference_mode():
            for i in range(len(source_audio_paths)):
                if model_name == "blaser_2_0_ref":
                    # Reference-based BLASER
                    score = blaser_model(
                        src=src_embs[i:i+1].to(device_obj),
                        ref=ref_embs[i:i+1].to(device_obj),
                        mt=mt_embs[i:i+1].to(device_obj),
                    ).item()
                else:
                    # QE-based BLASER (reference-free)
                    score = blaser_model(
                        src=src_embs[i:i+1].to(device_obj),
                        mt=mt_embs[i:i+1].to(device_obj),
                    ).item()
                scores.append(score)

        # Calculate corpus score
        corpus_score = sum(scores) / len(scores) if scores else 0.0

        # Prepare output
        result = {
            'corpus_score': corpus_score,
            'sentence_scores': scores,
            'model': model_name,
            'device': str(device_obj),
            'num_samples': len(scores),
            'source_lang': source_lang,
            'target_lang': target_lang,
        }

        # Save results
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Corpus score: {corpus_score:.4f}")

        print(json.dumps(result))  # Print to stdout for subprocess capture

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        error_result = {
            'error': str(e),
            'corpus_score': 0.0,
            'sentence_scores': [],
        }
        with open(output, 'w') as f:
            json.dump(error_result, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
