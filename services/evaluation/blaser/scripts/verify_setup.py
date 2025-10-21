#!/usr/bin/env python3
"""
Verify BLASER 2.0 and SONAR installation.

Tests:
1. Import all required libraries
2. Load BLASER models
3. Load SONAR encoders
4. Run simple prediction test
5. Report versions and capabilities
"""

import logging
import sys
from pathlib import Path

import click
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_imports():
    """Verify all required imports."""
    logger.info("Checking imports...")
    try:
        import fairseq2
        import sonar
        import numpy as np
        import torchaudio

        logger.info(f"✓ fairseq2: {fairseq2.__version__}")
        logger.info(f"✓ PyTorch: {torch.__version__}")
        logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA version: {torch.version.cuda}")
        logger.info(f"✓ sonar-space: installed")
        logger.info(f"✓ torchaudio: {torchaudio.__version__}")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_blaser_model(model_name: str = "blaser_2_0_qe"):
    """Test loading BLASER model."""
    logger.info(f"Testing BLASER model: {model_name}...")
    try:
        from sonar.models.blaser.loader import load_blaser_model

        model = load_blaser_model(model_name).eval()
        logger.info(f"✓ Successfully loaded {model_name}")

        # Test forward pass with dummy embeddings
        batch_size = 2
        embed_dim = 1024
        dummy_src = torch.randn(batch_size, embed_dim)
        dummy_mt = torch.randn(batch_size, embed_dim)

        if model_name == "blaser_2_0_ref":
            dummy_ref = torch.randn(batch_size, embed_dim)
            with torch.no_grad():
                scores = model(src=dummy_src, mt=dummy_mt, ref=dummy_ref)
        else:  # QE model
            with torch.no_grad():
                scores = model(src=dummy_src, mt=dummy_mt)

        logger.info(f"✓ Test prediction successful (scores shape: {scores.shape})")
        del model
        return True

    except Exception as e:
        logger.error(f"✗ BLASER model test failed: {e}")
        return False


def test_sonar_text_encoder():
    """Test SONAR text encoder."""
    logger.info("Testing SONAR text encoder...")
    try:
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

        encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
        )

        # Test encoding
        test_texts = ["Hello world", "This is a test"]
        embeddings = encoder.predict(test_texts, source_lang="eng_Latn")

        logger.info(f"✓ Text encoder loaded successfully")
        logger.info(f"✓ Embedding shape: {embeddings.shape}")
        logger.info(f"✓ Embedding dim: {embeddings.shape[1]}")

        del encoder
        return True

    except Exception as e:
        logger.error(f"✗ SONAR text encoder test failed: {e}")
        return False


def test_sonar_speech_encoder(lang: str = "eng"):
    """Test SONAR speech encoder."""
    logger.info(f"Testing SONAR speech encoder for {lang}...")
    try:
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
        import numpy as np

        # Language to encoder mapping
        lang_to_encoder = {
            'eng': 'sonar_speech_encoder_eng',
            'spa': 'sonar_speech_encoder_spa',
        }

        encoder_name = lang_to_encoder.get(lang, 'sonar_speech_encoder_eng')
        encoder = SpeechToEmbeddingModelPipeline(encoder=encoder_name)

        logger.info(f"✓ Speech encoder for {lang} loaded successfully")

        # Note: Can't test with actual audio without a file
        # Just verify it loads
        del encoder
        return True

    except Exception as e:
        logger.error(f"✗ SONAR speech encoder test failed: {e}")
        logger.error("  This might be expected if the model wasn't downloaded yet")
        return False


def report_capabilities():
    """Report supported languages and features."""
    logger.info("\nSupported Features:")
    logger.info("  - BLASER 2.0 QE (reference-free quality estimation)")
    logger.info("  - BLASER 2.0 REF (reference-based evaluation)")
    logger.info("  - SONAR text encoders (200+ languages)")
    logger.info("  - SONAR speech encoders (37 languages)")
    logger.info("\nCommon speech encoder languages:")
    logger.info("  eng, spa, fra, deu, ita, por, rus, cmn, jpn, kor,")
    logger.info("  arb, hin, tel, urd, vie, tha, ind, msa, tgl,")
    logger.info("  swh, afr, amh, yor, ibo, zul")


@click.command()
@click.option(
    '--blaser-model',
    default='blaser_2_0_qe',
    help='BLASER model to test (blaser_2_0_qe or blaser_2_0_ref)'
)
@click.option(
    '--test-speech',
    is_flag=True,
    help='Test speech encoder (requires downloaded models)'
)
@click.option(
    '--speech-lang',
    default='eng',
    help='Language for speech encoder test'
)
def main(blaser_model: str, test_speech: bool, speech_lang: str):
    """Verify BLASER 2.0 and SONAR setup."""

    logger.info("="*60)
    logger.info("BLASER 2.0 Setup Verification")
    logger.info("="*60)
    logger.info("")

    results = []

    # Check imports
    results.append(("Imports", check_imports()))
    logger.info("")

    # Test BLASER model
    results.append(("BLASER Model", test_blaser_model(blaser_model)))
    logger.info("")

    # Test SONAR text encoder
    results.append(("SONAR Text Encoder", test_sonar_text_encoder()))
    logger.info("")

    # Optionally test speech encoder
    if test_speech:
        results.append((f"SONAR Speech Encoder ({speech_lang})",
                       test_sonar_speech_encoder(speech_lang)))
        logger.info("")

    # Report capabilities
    report_capabilities()

    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("Verification Summary:")
    logger.info("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"  {test_name:30s} {status}")
        if not passed:
            all_passed = False

    logger.info("")

    if all_passed:
        logger.info("✓ All tests passed! BLASER environment is ready.")
        logger.info("")
        logger.info("You can now use evaluate.py for BLASER evaluation")
        sys.exit(0)
    else:
        logger.error("✗ Some tests failed. Please check the errors above.")
        logger.error("")
        logger.error("If models are missing, run:")
        logger.error("  python scripts/download_models.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
