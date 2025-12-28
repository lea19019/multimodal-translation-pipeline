#!/usr/bin/env python3
"""
Evaluate Pipeline Synthesis Samples

Runs evaluation metrics (BLEU, chrF, COMET, MCD, BLASER) on synthesized samples.

Usage:
    python evaluate_pipeline_samples.py --language efik --csv-path /path/to/predicted.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from text_metrics import TextMetrics
from comet_evaluator import CometEvaluator
from audio_metrics import AudioMetrics
from blaser_evaluator import BlaserEvaluator
from visualizations import create_metrics_comparison_chart, create_score_distribution, generate_html_report
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_csv_samples(csv_path: Path, nmt_csv_path: Path, language: str, audio_dir: Path = None):
    """
    Evaluate samples from a synthesis CSV file.
    
    Args:
        csv_path: Path to predicted synthesis CSV (predicted_*.csv)
        nmt_csv_path: Path to NMT predictions CSV (for ground truth)
        language: Language name (efik, igbo, swahili, xhosa)
        audio_dir: Directory containing synthesized audio files (auto-detected if None)
    """
    # Extract pipeline name and execution_id from CSV filename
    # e.g., "predicted_nllb_tgt_MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632.csv"
    # -> pipeline_name = "nllb_tgt_MULTILINGUAL_TRAINING"
    # -> execution_id = "MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632"
    csv_name = csv_path.stem  # Remove .csv
    execution_id = csv_name.replace("predicted_", "")
    
    if csv_name.startswith("predicted_"):
        # Remove "predicted_" prefix
        parts = csv_name.replace("predicted_", "").split("_")
        # Find where the checkpoint/timestamp starts (first part with digits)
        pipeline_parts = []
        checkpoint_start = -1
        for i, part in enumerate(parts):
            # Look for checkpoint name (usually starts with uppercase or has numbers)
            if (part.isupper() or any(c.isdigit() for c in part)) and len(part) > 3:
                # Check if this looks like a checkpoint (e.g., MULTILINGUAL, Src)
                if part in ['MULTILINGUAL', 'Src', 'Translate', 'Multilingual'] or part.startswith('MULTILINGUAL'):
                    checkpoint_start = i
                    break
            pipeline_parts.append(part)
        
        pipeline_name = "_".join(pipeline_parts)
    else:
        pipeline_name = "unknown"
    
    logger.info("="*70)
    logger.info(f"EVALUATING: {csv_path.name}")
    logger.info(f"PIPELINE: {pipeline_name}")
    logger.info(f"EXECUTION ID: {execution_id}")
    logger.info("="*70)
    
    # Load data
    logger.info(f"Loading predicted samples: {csv_path}")
    predicted_df = pd.read_csv(csv_path, sep='|')
    logger.info(f"  Rows: {len(predicted_df)}")
    
    logger.info(f"Loading ground truth NMT data: {nmt_csv_path}")
    nmt_df = pd.read_csv(nmt_csv_path, sep='|')
    logger.info(f"  Rows: {len(nmt_df)}")
    
    # Merge on segment_id
    logger.info("\nMerging data on segment_id...")
    merged_df = predicted_df.merge(
        nmt_df[['segment_id', 'ground_truth_tgt_text', 'src_text', 'predicted_tgt_text']],
        on='segment_id',
        how='left'
    )
    
    # Filter successful syntheses
    successful = merged_df[merged_df['success'] == True].copy()
    logger.info(f"Successful syntheses: {len(successful)} / {len(merged_df)}")
    
    if len(successful) == 0:
        logger.error("No successful syntheses to evaluate!")
        return None
    
    # Prepare data for evaluation
    predictions = successful['text'].tolist()
    references = successful['ground_truth_tgt_text'].tolist()
    sources = successful['src_text'].tolist()
    
    logger.info(f"\nEvaluating {len(predictions)} samples...")
    logger.info(f"Sample prediction: {predictions[0][:80]}...")
    logger.info(f"Sample reference:  {references[0][:80]}...")
    
    # Compute text metrics
    logger.info("\n" + "="*70)
    logger.info("TEXT METRICS")
    logger.info("="*70)
    
    # Initialize metrics
    text_metrics = TextMetrics(chrf_word_order=2)  # chrF++ with word order
    
    # BLEU
    logger.info("\n📊 Computing BLEU...")
    bleu_results = text_metrics.compute_bleu(
        hypotheses=predictions,
        references=[[ref] for ref in references]
    )
    logger.info(f"  Corpus BLEU: {bleu_results['corpus_score']:.2f}")
    logger.info(f"  Avg Sentence BLEU: {sum(bleu_results['sentence_scores'])/len(bleu_results['sentence_scores']):.2f}")
    
    # chrF++
    logger.info("\n📊 Computing chrF++...")
    chrf_results = text_metrics.compute_chrf(
        hypotheses=predictions,
        references=[[ref] for ref in references]
    )
    logger.info(f"  Corpus chrF++: {chrf_results['corpus_score']:.2f}")
    logger.info(f"  Avg Sentence chrF++: {sum(chrf_results['sentence_scores'])/len(chrf_results['sentence_scores']):.2f}")
    
    # COMET
    try:
        logger.info("\n📊 Computing COMET (SSA-COMET-QE for African languages)...")
        comet = CometEvaluator(model_name="McGill-NLP/ssa-comet-qe")
        comet_results = comet.evaluate(
            sources=sources,
            hypotheses=predictions,
            references=references
        )
        logger.info(f"  Corpus COMET: {comet_results['corpus_score']:.4f}")
        logger.info(f"  Avg Sentence COMET: {sum(comet_results['sentence_scores'])/len(comet_results['sentence_scores']):.4f}")
    except Exception as e:
        logger.warning(f"COMET evaluation failed: {e}")
        comet_results = None
    
    # Audio metrics (MCD and BLASER)
    mcd_results = None
    blaser_results = None
    
    if audio_dir is None:
        # Auto-detect audio directory from CSV path
        audio_dir = csv_path.parent / csv_path.stem
    
    if audio_dir.exists():
        logger.info("\n" + "="*70)
        logger.info("AUDIO METRICS")
        logger.info("="*70)
        logger.info(f"Audio directory: {audio_dir}")
        
        # Get reference audio directory
        ref_audio_dir = csv_path.parent / "processed_audio_normalized"
        
        if ref_audio_dir.exists():
            # Build lists of audio file paths
            generated_audio_paths = []
            reference_audio_paths = []
            
            for idx, row in successful.iterrows():
                segment_id = row['segment_id']
                user_id = row['user_id']
                
                # Map language to ISO code
                lang_iso = {'efik': 'efi', 'igbo': 'ibo', 'swahili': 'swa', 'xhosa': 'xho'}[language]
                
                # Generated audio: Segment=X_User=Y_Language=Z_pred.wav
                gen_audio = audio_dir / f"Segment={segment_id}_User={user_id}_Language={lang_iso}_pred.wav"
                
                # Reference audio: Segment=X_User=Y_Language=Z.wav
                ref_audio = ref_audio_dir / f"Segment={segment_id}_User={user_id}_Language={lang_iso}.wav"
                
                if gen_audio.exists() and ref_audio.exists():
                    generated_audio_paths.append(gen_audio)
                    reference_audio_paths.append(ref_audio)
                else:
                    if not gen_audio.exists():
                        logger.warning(f"  Missing generated audio: {gen_audio.name}")
                    if not ref_audio.exists():
                        logger.warning(f"  Missing reference audio: {ref_audio.name}")
            
            logger.info(f"Found {len(generated_audio_paths)} audio pairs")
            
            if generated_audio_paths:
                # MCD
                try:
                    logger.info("\n🎵 Computing MCD (Mel-Cepstral Distance)...")
                    audio_metrics = AudioMetrics()
                    mcd_results = audio_metrics.compute_mcd_batch(
                        generated_audio_paths,
                        reference_audio_paths
                    )
                    logger.info(f"  Mean MCD: {mcd_results['mean_mcd']:.2f} (lower is better)")
                    logger.info(f"  Std MCD:  {mcd_results['std_mcd']:.2f}")
                    logger.info(f"  Min MCD:  {mcd_results['min_mcd']:.2f}")
                    logger.info(f"  Max MCD:  {mcd_results['max_mcd']:.2f}")
                    logger.info(f"  Successful: {mcd_results['num_successful']}/{mcd_results['num_samples']}")
                except Exception as e:
                    logger.warning(f"MCD evaluation failed: {e}")
                    mcd_results = None
                
                # BLASER
                try:
                    logger.info("\n🎵 Computing BLASER 2.0 (Speech-to-Speech Quality)...")
                    
                    # Map language to SONAR code
                    lang_codes = {
                        'efik': 'efi_Latn',
                        'igbo': 'ibo_Latn', 
                        'swahili': 'swh_Latn',
                        'xhosa': 'xho_Latn'
                    }
                    target_lang = lang_codes[language]
                    
                    # For BLASER, we need source audio too (use reference audio for now)
                    source_audio_paths = reference_audio_paths.copy()
                    
                    blaser = BlaserEvaluator(model_name="blaser_2_0_qe")
                    blaser_results = blaser.evaluate(
                        source_audio_paths=source_audio_paths,
                        target_audio_paths=generated_audio_paths,
                        reference_texts=references,
                        source_texts=sources,
                        source_lang=target_lang,  # Same language audio
                        target_lang=target_lang
                    )
                    logger.info(f"  Corpus BLASER: {blaser_results['corpus_score']:.4f} (0-5, higher is better)")
                    logger.info(f"  Avg Sentence BLASER: {sum(blaser_results['sentence_scores'])/len(blaser_results['sentence_scores']):.4f}")
                except Exception as e:
                    logger.warning(f"BLASER evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    blaser_results = None
        else:
            logger.warning(f"Reference audio directory not found: {ref_audio_dir}")
    else:
        logger.warning(f"Audio directory not found: {audio_dir}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*70)
    logger.info(f"File: {csv_path.name}")
    logger.info(f"Language: {language}")
    logger.info(f"Samples: {len(successful)}")
    logger.info(f"\nText Metrics:")
    logger.info(f"  BLEU:   {bleu_results['corpus_score']:6.2f}")
    logger.info(f"  chrF++: {chrf_results['corpus_score']:6.2f}")
    if comet_results:
        logger.info(f"  COMET:  {comet_results['corpus_score']:6.4f}")
    if mcd_results:
        logger.info(f"\nAudio Metrics:")
        logger.info(f"  MCD:    {mcd_results['mean_mcd']:6.2f} ± {mcd_results['std_mcd']:.2f}")
    if blaser_results:
        logger.info(f"  BLASER: {blaser_results['corpus_score']:6.4f}")
    logger.info("="*70)
    
    return {
        'csv_path': str(csv_path),
        'pipeline_name': pipeline_name,
        'execution_id': execution_id,
        'language': language,
        'n_samples': len(successful),
        'bleu': bleu_results,
        'chrf': chrf_results,
        'comet': comet_results,
        'mcd': mcd_results,
        'blaser': blaser_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate synthesized samples from pipeline testing"
    )
    
    parser.add_argument(
        '--language',
        type=str,
        required=True,
        choices=['efik', 'igbo', 'swahili', 'xhosa'],
        help='Full language name'
    )
    parser.add_argument(
        '--csv-path',
        type=Path,
        help='Path to predicted CSV file (if not provided, will find all in language dir)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('/home/vacl2/multimodal_translation/services/data/languages'),
        help='Base data directory'
    )
    
    args = parser.parse_args()
    
    # Get language directory
    lang_dir = args.data_dir / args.language
    if not lang_dir.exists():
        logger.error(f"Language directory not found: {lang_dir}")
        return 1
    
    # Get NMT predictions CSV (ground truth)
    nmt_csv = lang_dir / "nmt_predictions_multilang_finetuned_final.csv"
    if not nmt_csv.exists():
        logger.error(f"NMT predictions CSV not found: {nmt_csv}")
        return 1
    
    # Find predicted CSV files
    if args.csv_path:
        csv_files = [args.csv_path]
    else:
        csv_files = sorted(lang_dir.glob("predicted_*.csv"))
    
    if not csv_files:
        logger.error(f"No predicted CSV files found in {lang_dir}")
        return 1
    
    logger.info(f"Found {len(csv_files)} predicted CSV files to evaluate")
    
    # Evaluate each CSV
    all_results = []
    for csv_file in csv_files:
        try:
            result = evaluate_csv_samples(csv_file, nmt_csv, args.language)
            if result:
                all_results.append(result)
                
                # Save results organized by execution_id/pipeline_name
                execution_id = result['execution_id']
                pipeline_name = result['pipeline_name']
                
                # Structure: evaluation_results/{execution_id}/{pipeline_name}/
                eval_dir = lang_dir / "evaluation_results" / execution_id / pipeline_name
                eval_dir.mkdir(parents=True, exist_ok=True)
                
                # Save JSON results
                results_json = eval_dir / "metrics.json"
                with open(results_json, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                logger.info(f"\n📊 Results saved to: {results_json}")
                
                # Generate visualizations for this pipeline
                try:
                    # Aggregate scores for comparison chart
                    aggregate_scores = {}
                    if result['bleu']:
                        aggregate_scores['bleu'] = result['bleu']['corpus_score']
                    if result['chrf']:
                        aggregate_scores['chrf'] = result['chrf']['corpus_score']
                    if result['comet']:
                        aggregate_scores['comet'] = result['comet']['corpus_score']
                    if result['mcd']:
                        aggregate_scores['mcd'] = result['mcd']['mean_mcd']
                    if result['blaser']:
                        aggregate_scores['blaser'] = result['blaser']['corpus_score']
                    
                    if aggregate_scores:
                        # Create metrics comparison chart
                        viz_path = eval_dir / "metrics_comparison.png"
                        create_metrics_comparison_chart(aggregate_scores, viz_path)
                        logger.info(f"  ✓ Metrics comparison: {viz_path}")
                        
                        # Create distribution plots for each metric
                        for metric_name in ['bleu', 'chrf', 'comet', 'mcd', 'blaser']:
                            metric_data = result.get(metric_name)
                            if metric_data and 'sentence_scores' in metric_data:
                                scores = metric_data['sentence_scores']
                                if scores:
                                    dist_path = eval_dir / f"{metric_name}_distribution.png"
                                    create_score_distribution(scores, metric_name.upper(), dist_path)
                                    logger.info(f"  ✓ {metric_name.upper()} distribution: {dist_path}")
                    
                    logger.info(f"\n📊 Visualizations saved to: {eval_dir}")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate visualizations: {e}")
            
            print("\n")
        except Exception as e:
            logger.error(f"Failed to evaluate {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary of all evaluations
    if all_results:
        logger.info("\n" + "="*70)
        
        # Group by execution_id
        by_execution = {}
        for result in all_results:
            exec_id = result['execution_id']
            if exec_id not in by_execution:
                by_execution[exec_id] = []
            by_execution[exec_id].append(result)
        
        logger.info(f"\nResults by Execution:")
        for exec_id, results in by_execution.items():
            logger.info(f"\n  Execution: {exec_id}")
            for result in results:
                pipeline_name = result['pipeline_name']
                logger.info(f"    Pipeline: {pipeline_name}")
                logger.info(f"      Samples: {result['n_samples']}")
                metrics_line = f"      BLEU: {result['bleu']['corpus_score']:.2f}, chrF++: {result['chrf']['corpus_score']:.2f}"
                if result['blaser']:
                    metrics_line += f", BLASER: {result['blaser']['corpus_score']:.4f}"
                logger.info(metrics_line)
        
        # Save combined results
        combined_dir = args.data_dir / args.language / "evaluation_results"
        combined_dir.mkdir(exist_ok=True)
        combined_json = combined_dir / "all_results.json"
        with open(combined_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\n📊 All results saved to: {combined_json}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
