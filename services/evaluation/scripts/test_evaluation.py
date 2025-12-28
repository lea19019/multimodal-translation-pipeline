#!/usr/bin/env python3
"""
Test Evaluation Script

Quick test of evaluation metrics on existing synthesized samples.
Tests both text metrics (BLEU, chrF, COMET) and audio metrics (MCD, BLASER).

Usage:
    python test_evaluation.py --language efi --descriptor nllb_tgt
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add evaluation module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_text_evaluation(predicted_df, ground_truth_df, language):
    """Test text-based evaluation metrics."""
    logger.info("\n=== Testing Text Evaluation Metrics ===")
    
    try:
        from evaluation import evaluate_text
        
        # Prepare data
        predictions = predicted_df['text'].tolist()
        references = ground_truth_df['ground_truth_tgt_text'].tolist()
        
        # Ensure same number of samples
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
        
        logger.info(f"Evaluating {len(predictions)} samples")
        logger.info(f"Sample prediction: {predictions[0][:50]}...")
        logger.info(f"Sample reference: {references[0][:50]}...")
        
        # Run evaluation
        results = evaluate_text(
            predictions=predictions,
            references=references,
            language=language
        )
        
        logger.info("\n✓ Text Evaluation Results:")
        for metric, score in results.items():
            logger.info(f"  {metric}: {score:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"✗ Text evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_audio_evaluation(predicted_df, ground_truth_audio_dir, audio_dir, language):
    """Test audio-based evaluation metrics."""
    logger.info("\n=== Testing Audio Evaluation Metrics ===")
    
    try:
        # For now, just check that audio files exist
        audio_files = list(audio_dir.glob("*.wav"))
        logger.info(f"Found {len(audio_files)} synthesized audio files")
        
        if audio_files:
            logger.info(f"Sample audio: {audio_files[0].name}")
            logger.info(f"File size: {audio_files[0].stat().st_size / 1024:.1f} KB")
        
        # TODO: Implement actual MCD and BLASER evaluation
        logger.info("\n⚠ Audio metrics (MCD, BLASER) not yet implemented")
        logger.info("  This would compare:")
        logger.info(f"  - Synthesized: {audio_dir}")
        logger.info(f"  - Ground truth: {ground_truth_audio_dir}")
        
        return {
            'n_audio_files': len(audio_files),
            'mcd': None,
            'blaser': None
        }
        
    except Exception as e:
        logger.error(f"✗ Audio evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test evaluation metrics on synthesized samples"
    )
    
    parser.add_argument(
        '--language',
        type=str,
        required=True,
        choices=['efi', 'ibo', 'swa', 'xho'],
        help='Language ISO code'
    )
    parser.add_argument(
        '--descriptor',
        type=str,
        default='nllb_tgt',
        help='Pipeline descriptor (e.g., nllb_tgt, src, custom_lang)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632',
        help='Model checkpoint name'
    )
    
    args = parser.parse_args()
    
    # Map ISO to full language name
    lang_map = {
        'efi': 'efik',
        'ibo': 'igbo', 
        'swa': 'swahili',
        'xho': 'xhosa'
    }
    full_lang = lang_map[args.language]
    
    # Paths
    data_dir = Path('/home/vacl2/multimodal_translation/services/data/languages') / full_lang
    predicted_dir = data_dir / f"predicted_{args.descriptor}_{args.checkpoint}"
    predicted_csv = data_dir / f"predicted_{args.descriptor}_{args.checkpoint}.csv"
    nmt_csv = data_dir / "nmt_predictions_multilang_finetuned_final.csv"
    
    logger.info("="*70)
    logger.info("EVALUATION TEST")
    logger.info("="*70)
    logger.info(f"Language: {full_lang} ({args.language})")
    logger.info(f"Descriptor: {args.descriptor}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Predicted CSV: {predicted_csv}")
    logger.info(f"Audio directory: {predicted_dir}")
    logger.info("")
    
    # Check files exist
    if not predicted_csv.exists():
        logger.error(f"Predicted CSV not found: {predicted_csv}")
        return 1
    
    if not nmt_csv.exists():
        logger.error(f"Ground truth CSV not found: {nmt_csv}")
        return 1
    
    # Load data
    logger.info("Loading data...")
    predicted_df = pd.read_csv(predicted_csv, sep='|')
    ground_truth_df = pd.read_csv(nmt_csv, sep='|')
    
    logger.info(f"Predicted samples: {len(predicted_df)}")
    logger.info(f"Ground truth samples: {len(ground_truth_df)}")
    logger.info(f"Predicted columns: {predicted_df.columns.tolist()}")
    
    # Merge on segment_id to align samples
    logger.info("\nMerging predicted and ground truth data...")
    merged_df = predicted_df.merge(
        ground_truth_df[['segment_id', 'ground_truth_tgt_text', 'src_text']],
        on='segment_id',
        how='left'
    )
    logger.info(f"Merged samples: {len(merged_df)}")
    
    # Filter only successful syntheses
    successful = merged_df[merged_df['success'] == True]
    logger.info(f"Successful syntheses: {len(successful)}")
    
    if len(successful) == 0:
        logger.error("No successful syntheses found!")
        return 1
    
    # Test text evaluation
    text_results = test_text_evaluation(
        successful,
        successful,
        args.language
    )
    
    # Test audio evaluation
    audio_results = test_audio_evaluation(
        successful,
        ground_truth_audio_dir=data_dir / "tgt_audio",
        audio_dir=predicted_dir,
        language=args.language
    )
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION TEST COMPLETE")
    logger.info("="*70)
    
    if text_results:
        logger.info("\n✓ Text Evaluation: SUCCESS")
        logger.info(f"  Samples evaluated: {len(successful)}")
        for metric, score in text_results.items():
            logger.info(f"  {metric}: {score:.4f}")
    else:
        logger.error("\n✗ Text Evaluation: FAILED")
    
    if audio_results:
        logger.info("\n⚠ Audio Evaluation: PARTIAL (file check only)")
        logger.info(f"  Audio files found: {audio_results['n_audio_files']}")
    else:
        logger.error("\n✗ Audio Evaluation: FAILED")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
