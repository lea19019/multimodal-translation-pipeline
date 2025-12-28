#!/usr/bin/env python3
"""
Quick test to verify BLASER works on a small sample.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from blaser_evaluator import BlaserEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_blaser_small_sample():
    """Test BLASER on just 5 samples from pipeline 1."""

    # Load metadata from pipeline 1
    metadata_path = Path("/home/vacl2/multimodal_translation/services/data/languages/efik") / \
                    "predicted_nllb_tgt_MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632" / \
                    "metadata.csv"

    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        return

    # Load metadata
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} samples from metadata")

    # Take only first 5 samples
    df_small = df.head(5)
    logger.info(f"Testing BLASER on {len(df_small)} samples")

    # Get base directory
    base_dir = metadata_path.parent

    # Prepare data
    source_audio = [base_dir / row['source_audio_path'] for _, row in df_small.iterrows()]
    target_audio = [base_dir / row['synthesized_audio_path'] for _, row in df_small.iterrows()]
    reference_texts = df_small['target_text'].tolist()
    source_texts = df_small['source_text'].tolist()

    # Check files exist
    for i, (src, tgt) in enumerate(zip(source_audio, target_audio)):
        if not src.exists():
            logger.error(f"Source audio {i+1} not found: {src}")
            return
        if not tgt.exists():
            logger.error(f"Target audio {i+1} not found: {tgt}")
            return

    logger.info("All audio files found ✓")

    # Initialize BLASER
    logger.info("Initializing BLASER evaluator...")
    evaluator = BlaserEvaluator(model_name='blaser_2_0_qe')

    # Run evaluation
    logger.info("Running BLASER evaluation (this may take a few minutes)...")
    result = evaluator.evaluate(
        source_audio_paths=source_audio,
        target_audio_paths=target_audio,
        reference_texts=reference_texts,
        source_texts=source_texts,
        source_lang='eng_Latn',
        target_lang='efi_Latn'  # Efik
    )

    # Print results
    logger.info("\n" + "="*60)
    logger.info("BLASER TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Corpus Score: {result['corpus_score']:.4f}")
    logger.info(f"Sample Count: {len(result['sentence_scores'])}")
    logger.info(f"Sentence Scores: {result['sentence_scores']}")
    logger.info(f"Signature: {result.get('signature', 'N/A')}")
    if 'error' in result:
        logger.error(f"Error: {result['error']}")
    logger.info("="*60 + "\n")

    # Check if it worked
    if result['corpus_score'] > 0:
        logger.info("✓ BLASER is working correctly!")
        return True
    else:
        logger.error("✗ BLASER returned 0 score - something is wrong")
        return False


if __name__ == '__main__':
    success = test_blaser_small_sample()
    sys.exit(0 if success else 1)
