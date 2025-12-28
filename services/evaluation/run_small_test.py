#!/usr/bin/env python3
"""
Small-sample test for pipeline comparison system.

Tests with just 10 samples to verify:
1. All metrics compute correctly
2. BLASER actually works
3. COMET loads properly
4. Results make sense
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add module directories to path
sys.path.insert(0, str(Path(__file__).parent / 'config'))
sys.path.insert(0, str(Path(__file__).parent / 'orchestrator'))
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from pipeline_config import get_synthesis_path, LANGUAGES
import pandas as pd
from text_metrics import TextMetrics
from comet_evaluator import CometEvaluator
from audio_metrics import AudioMetrics
from blaser_evaluator import BlaserEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def test_pipeline_small_sample(pipeline_id='pipeline_1', language='efik', n_samples=10):
    """
    Test a single pipeline with small sample size.

    Args:
        pipeline_id: Pipeline to test
        language: Language to test
        n_samples: Number of samples to use (default 10)
    """

    print("="*70)
    print(f"SMALL SAMPLE TEST: {pipeline_id} × {language} ({n_samples} samples)")
    print("="*70 + "\n")

    # Get paths
    paths = get_synthesis_path(pipeline_id, language)
    pipeline = paths['pipeline']
    csv_path = paths['csv_path']
    nmt_csv_path = paths['nmt_csv_path']
    audio_dir = paths['audio_dir']
    ref_audio_dir = paths['ref_audio_dir']
    iso_code = paths['iso_code']

    logger.info(f"Pipeline: {pipeline['name']}")
    logger.info(f"CSV: {csv_path.name}")
    logger.info(f"NMT CSV: {nmt_csv_path.name}")

    # Load data
    predicted_df = pd.read_csv(csv_path, sep='|')
    nmt_df = pd.read_csv(nmt_csv_path, sep='|')

    logger.info(f"Synthesis CSV rows: {len(predicted_df)}")
    logger.info(f"NMT CSV rows: {len(nmt_df)}")

    # Merge
    merged_df = predicted_df.merge(
        nmt_df[['segment_id', 'ground_truth_tgt_text', 'src_text', 'predicted_tgt_text']],
        on='segment_id',
        how='left'
    )

    logger.info(f"Merged rows: {len(merged_df)}")

    # Check for duplicates
    duplicates = merged_df['segment_id'].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"⚠️  Found {duplicates} duplicate segment_ids!")

    # Filter successful and remove duplicates
    successful = merged_df[merged_df['success'] == True].copy()
    logger.info(f"Successful syntheses before dedup: {len(successful)}")

    # Remove duplicates by segment_id (keep first occurrence)
    successful = successful.drop_duplicates(subset=['segment_id'], keep='first')
    logger.info(f"After removing duplicates: {len(successful)}")

    # Take only first N samples
    small_sample = successful.head(n_samples)
    logger.info(f"\n📊 Using {len(small_sample)} samples for testing\n")

    # Prepare data
    predictions = small_sample['text'].tolist()
    references = small_sample['ground_truth_tgt_text'].tolist()
    sources = small_sample['src_text'].tolist()

    # Audio paths (use segment_id + user_id format)
    source_audio = []
    target_audio = []

    for _, row in small_sample.iterrows():
        segment_id = row['segment_id']
        user_id = row['user_id']

        # Generated audio
        gen_audio = audio_dir / f"Segment={segment_id}_User={user_id}_Language={iso_code}_pred.wav"
        target_audio.append(gen_audio)

        # Reference audio
        ref_audio = ref_audio_dir / f"Segment={segment_id}_User={user_id}_Language={iso_code}.wav"
        source_audio.append(ref_audio)

    # Verify audio files exist
    missing = []
    for i, (src, tgt) in enumerate(zip(source_audio, target_audio)):
        if not src.exists():
            missing.append(f"Source {i}: {src}")
        if not tgt.exists():
            missing.append(f"Target {i}: {tgt}")

    if missing:
        logger.error("Missing audio files:")
        for m in missing:
            logger.error(f"  {m}")
        return None

    logger.info("✓ All audio files found")

    # Results dict
    results = {}

    # =========================================================================
    # TEXT METRICS (if pipeline uses NMT)
    # =========================================================================
    if pipeline['uses_nmt']:
        print("\n" + "="*70)
        print("TEXT METRICS")
        print("="*70)

        text_metrics = TextMetrics(chrf_word_order=2)

        # BLEU
        logger.info("\n📊 Computing BLEU...")
        try:
            bleu_result = text_metrics.compute_bleu(
                hypotheses=predictions,
                references=[[ref] for ref in references]
            )
            results['bleu'] = bleu_result['corpus_score']
            logger.info(f"  ✓ BLEU: {bleu_result['corpus_score']:.2f} (0-100, higher is better)")
        except Exception as e:
            logger.error(f"  ✗ BLEU failed: {e}")
            results['bleu'] = None

        # chrF++
        logger.info("\n📊 Computing chrF++...")
        try:
            chrf_result = text_metrics.compute_chrf(
                hypotheses=predictions,
                references=[[ref] for ref in references]
            )
            results['chrf'] = chrf_result['corpus_score']
            logger.info(f"  ✓ chrF++: {chrf_result['corpus_score']:.2f} (0-100, higher is better)")
        except Exception as e:
            logger.error(f"  ✗ chrF++ failed: {e}")
            results['chrf'] = None

        # COMET
        logger.info("\n📊 Computing COMET...")
        try:
            # Try different model names
            model_names = [
                'Unbabel/wmt22-comet-da',  # Generic COMET model
                'Unbabel/XCOMET-XL',       # Multilingual COMET
                'McGill-NLP/ssa-comet-qe'  # SSA-specific (may not work)
            ]

            comet_result = None
            for model_name in model_names:
                try:
                    logger.info(f"  Trying model: {model_name}")
                    comet_eval = CometEvaluator(model_name=model_name)
                    comet_result = comet_eval.evaluate(
                        sources=sources,
                        hypotheses=predictions,
                        references=references
                    )
                    results['comet'] = comet_result['corpus_score']
                    logger.info(f"  ✓ COMET: {comet_result['corpus_score']:.4f} (0-1, higher is better)")
                    break
                except Exception as e:
                    logger.warning(f"  Model {model_name} failed: {e}")
                    continue

            if comet_result is None:
                logger.error("  ✗ All COMET models failed")
                results['comet'] = None

        except Exception as e:
            logger.error(f"  ✗ COMET failed: {e}")
            results['comet'] = None

    # =========================================================================
    # AUDIO METRICS
    # =========================================================================
    print("\n" + "="*70)
    print("AUDIO METRICS")
    print("="*70)

    # MCD
    logger.info("\n📊 Computing MCD...")
    try:
        audio_metrics = AudioMetrics()
        mcd_result = audio_metrics.compute_mcd_from_audio(
            predicted_audio=target_audio,
            reference_audio=source_audio
        )
        results['mcd'] = mcd_result['mean_mcd']
        logger.info(f"  ✓ MCD: {mcd_result['mean_mcd']:.2f} dB (lower is better)")
    except Exception as e:
        logger.error(f"  ✗ MCD failed: {e}")
        results['mcd'] = None

    # BLASER
    logger.info("\n📊 Computing BLASER (may take a few minutes on CPU)...")
    try:
        blaser_eval = BlaserEvaluator(model_name='blaser_2_0_qe')
        blaser_result = blaser_eval.evaluate(
            source_audio_paths=source_audio,
            target_audio_paths=target_audio,
            reference_texts=references,
            source_texts=sources,
            source_lang='eng_Latn',
            target_lang=LANGUAGES[language]['iso_code']
        )
        results['blaser'] = blaser_result['corpus_score']

        if 'error' in blaser_result:
            logger.error(f"  ✗ BLASER error: {blaser_result['error']}")
        elif blaser_result['corpus_score'] > 0:
            logger.info(f"  ✓ BLASER: {blaser_result['corpus_score']:.4f} (0-5, higher is better)")
        else:
            logger.error("  ✗ BLASER returned 0.0000 (timeout or error)")

    except Exception as e:
        logger.error(f"  ✗ BLASER failed: {e}")
        results['blaser'] = None

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\nPipeline: {pipeline['name']}")
    print(f"Language: {language}")
    print(f"Samples:  {n_samples}\n")

    if pipeline['uses_nmt']:
        print("Text Metrics:")
        print(f"  BLEU:   {results.get('bleu', 'N/A'):.2f}" if results.get('bleu') else "  BLEU:   Failed")
        print(f"  chrF++: {results.get('chrf', 'N/A'):.2f}" if results.get('chrf') else "  chrF++: Failed")
        print(f"  COMET:  {results.get('comet', 'N/A'):.4f}" if results.get('comet') else "  COMET:  Failed")
        print()

    print("Audio Metrics:")
    print(f"  MCD:    {results.get('mcd', 'N/A'):.2f} dB" if results.get('mcd') is not None else "  MCD:    Failed")
    print(f"  BLASER: {results.get('blaser', 'N/A'):.4f}" if results.get('blaser') is not None else "  BLASER: Failed")

    print("\n" + "="*70 + "\n")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run small-sample pipeline test')
    parser.add_argument('--pipeline', default='pipeline_1', help='Pipeline ID')
    parser.add_argument('--language', default='efik', help='Language')
    parser.add_argument('--n-samples', type=int, default=10, help='Number of samples')

    args = parser.parse_args()

    results = test_pipeline_small_sample(
        pipeline_id=args.pipeline,
        language=args.language,
        n_samples=args.n_samples
    )

    if results:
        # Check if all metrics worked
        failures = [k for k, v in results.items() if v is None or v == 0]
        if not failures:
            logger.info("✓ All metrics computed successfully!")
            sys.exit(0)
        else:
            logger.error(f"✗ Some metrics failed: {failures}")
            sys.exit(1)
    else:
        logger.error("✗ Test failed")
        sys.exit(1)
