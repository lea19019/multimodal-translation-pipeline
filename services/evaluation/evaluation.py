#!/usr/bin/env python3
"""
Main evaluation script for multimodal translation evaluation.

Evaluates translations across different modalities using appropriate metrics.

This is a simplified entry point that delegates to the CLI module.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import yaml

from scripts.data_loader import load_samples, load_predictions
from pipeline import EvaluationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@click.command()
@click.option('--type', '-t', 'translation_type',
              type=click.Choice(['text_to_text', 'audio_to_text', 'text_to_audio', 'audio_to_audio', 'all']),
              help='Translation type to evaluate')
@click.option('--data-dir', '-d', type=click.Path(exists=True),
              help='Path to data directory (required unless using --config)')
@click.option('--output-dir', '-o', type=click.Path(),
              default='./results', help='Output directory for results')
@click.option('--samples', '-s', multiple=True,
              help='Specific sample UUIDs to evaluate')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to YAML config file')
@click.option('--run-id', help='Custom run ID')
@click.option('--metrics', '-m', multiple=True,
              type=click.Choice(['bleu', 'chrf', 'comet', 'mcd', 'blaser']),
              help='Specific metrics to compute (auto-detect if not specified)')
@click.option('--mode', type=click.Choice(['ground_truth', 'predictions']),
              default='ground_truth',
              help='Evaluation mode: ground_truth (original) or predictions (evaluate model outputs)')
@click.option('--language', type=click.Choice(['efik', 'igbo', 'swahili', 'xhosa']),
              help='Single language to evaluate (only for predictions mode)')
@click.option('--limit', type=int, default=None,
              help='Limit number of samples to evaluate (for testing)')
@click.option('--nmt-model', type=str, required=False,
              help='NMT model name (required for predictions mode, e.g., multilang_finetuned_final)')
@click.option('--tts-model', type=str, required=False,
              help='TTS model name (required for predictions mode, e.g., MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632)')
@click.option('--execution-id', type=str, default=None,
              help='Execution ID (auto-generated if not provided)')
def main(
    translation_type: Optional[str],
    data_dir: str,
    output_dir: str,
    samples: tuple,
    config: Optional[str],
    run_id: Optional[str],
    metrics: tuple,
    mode: str,
    language: Optional[str],
    limit: Optional[int],
    nmt_model: Optional[str],
    tts_model: Optional[str],
    execution_id: Optional[str],
):
    """
    Evaluate multimodal translation system.

    Examples:

        \\b
        # Evaluate text-to-text translations
        python evaluation.py --type text_to_text --data-dir ../data/app_evaluation/text_to_text

        \\b
        # Evaluate all translation types
        python evaluation.py --type all --data-dir ../data/app_evaluation

        \\b
        # Evaluate specific samples
        python evaluation.py --type text_to_audio --samples uuid1 uuid2 --data-dir ../data/app_evaluation/text_to_audio

        \\b
        # Use config file
        python evaluation.py --config evaluation_config.yaml
    """
    try:
        # Load config if provided
        if config:
            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)
            translation_type = translation_type or config_data.get('translation_type')
            data_dir = data_dir or config_data.get('data_dir')
            output_dir = output_dir or config_data.get('output_dir', './results')
            run_id = run_id or config_data.get('run_id')
            if not metrics and 'metrics' in config_data:
                metrics = tuple(config_data['metrics'])

        # Validate required parameters
        if not data_dir:
            click.echo("Error: --data-dir is required (either as argument or in config file)")
            sys.exit(1)

        # Validate predictions mode requirements
        if mode == 'predictions':
            if not nmt_model:
                click.echo("Error: --nmt-model is required for predictions mode")
                sys.exit(1)
            if not tts_model:
                click.echo("Error: --tts-model is required for predictions mode")
                sys.exit(1)

        # Auto-generate execution ID if not provided (for predictions mode)
        if mode == 'predictions' and not execution_id:
            execution_id = datetime.now().strftime('eval_%Y%m%d_%H%M%S')
            logger.info(f"Auto-generated execution ID: {execution_id}")
        elif execution_id:
            logger.info(f"Using provided execution ID: {execution_id}")

        # Update output directory structure for predictions mode
        if mode == 'predictions' and execution_id:
            # Results go to: services/evaluation/results/{execution_id}/
            # Each language will have its own subdirectory
            base_results_dir = Path(output_dir) / 'results' / execution_id
            base_results_dir.mkdir(parents=True, exist_ok=True)

            # Manifest file path
            manifest_path = base_results_dir / 'manifest.json'

            # Initialize manifest if it doesn't exist
            if not manifest_path.exists():
                manifest = {
                    'execution_id': execution_id,
                    'timestamp': datetime.now().isoformat(),
                    'nmt_model': nmt_model,
                    'tts_model': tts_model,
                    'metrics': list(metrics) if metrics else [],
                    'languages': {},
                    'total_samples': 0,
                    'total_valid_samples': 0,
                }
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
                logger.info(f"Initialized manifest: {manifest_path}")

        # Initialize pipeline
        # For predictions mode, we'll create a new pipeline per language
        # For ground truth mode, create it once
        if mode != 'predictions':
            pipeline = EvaluationPipeline(output_dir=output_dir, run_id=run_id)
            logger.info(f"Starting evaluation with run_id: {pipeline.run_id}")

        # Determine translation types to evaluate
        if translation_type == 'all':
            types_to_eval = ['text_to_text', 'audio_to_text', 'text_to_audio', 'audio_to_audio']
        else:
            types_to_eval = [translation_type] if translation_type else [None]

        # Evaluate each type
        all_results = []

        # For predictions mode, evaluate by language instead of by type
        if mode == 'predictions':
            _evaluate_predictions_mode(
                language, languages_to_eval=['efik', 'igbo', 'swahili', 'xhosa'],
                base_results_dir=base_results_dir, manifest_path=manifest_path,
                execution_id=execution_id, run_id=run_id, data_dir=data_dir,
                nmt_model=nmt_model, tts_model=tts_model, limit=limit,
                metrics=metrics, all_results=all_results
            )
        else:
            # Ground truth mode (original behavior)
            _evaluate_ground_truth_mode(
                types_to_eval, pipeline, data_dir, samples, limit,
                metrics, all_results
            )

        if not all_results:
            logger.error("No results generated")
            sys.exit(1)

        click.echo(f"\n{'='*60}")
        click.echo("ALL EVALUATIONS COMPLETE")
        click.echo(f"{'='*60}")
        click.echo(f"Results directory: {pipeline.results_dir if mode != 'predictions' else base_results_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


def _evaluate_predictions_mode(
    language, languages_to_eval, base_results_dir, manifest_path,
    execution_id, run_id, data_dir, nmt_model, tts_model, limit, metrics, all_results
):
    """Evaluate in predictions mode (by language)."""
    # Determine which languages to evaluate
    if language:
        languages_to_eval = [language]

    for lang in languages_to_eval:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating predictions for {lang.upper()}")
        logger.info(f"{'='*60}")

        # Create a new pipeline for this language with proper directory structure
        if execution_id:
            lang_results_dir = base_results_dir / lang
            # Create pipeline with language-specific directory, no nested run_id
            pipeline = EvaluationPipeline(
                output_dir=lang_results_dir,
                run_id=run_id,
                use_run_id_subdir=False  # Don't create nested run_id directory
            )
            logger.info(f"Results will be saved to: {lang_results_dir}")
        else:
            # Fallback for predictions without execution_id
            pipeline = EvaluationPipeline(output_dir=Path('./results'), run_id=run_id)

        # Load predictions with model names
        loaded_samples, errors = load_predictions(
            data_dir=data_dir,
            language=lang,
            nmt_model=nmt_model,
            tts_model=tts_model,
            translation_type='audio_to_audio',  # Default for predictions
        )

        if not loaded_samples:
            logger.warning(f"No prediction samples loaded for {lang}")
            continue

        # Apply limit if specified
        if limit:
            logger.info(f"Limiting evaluation to {limit} samples")
            loaded_samples = loaded_samples[:limit]

        # Evaluate
        results = pipeline.evaluate_samples(
            loaded_samples,
            metrics=list(metrics) if metrics else None,
        )

        if results:
            # Save results
            pipeline.save_results(results)

            # Generate visualizations
            pipeline.generate_visualizations(results)

            all_results.append(results)

            # Update manifest with this language's results
            if execution_id:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)

                manifest['languages'][lang] = {
                    'total_samples': results['total_samples'],
                    'valid_samples': results['valid_samples'],
                    'aggregate_scores': results.get('aggregate_scores', {}),
                }
                manifest['total_samples'] += results['total_samples']
                manifest['total_valid_samples'] += results['valid_samples']

                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
                logger.info(f"Updated manifest: {manifest_path}")

            # Print summary
            click.echo(f"\n{'='*60}")
            click.echo(f"EVALUATION COMPLETE - {lang}")
            click.echo(f"{'='*60}")
            click.echo(f"Execution ID: {execution_id}")
            click.echo(f"Run ID: {pipeline.run_id}")
            click.echo(f"Samples evaluated: {results['valid_samples']}/{results['total_samples']}")
            click.echo(f"\nAggregate Scores:")
            for metric, score in results.get('aggregate_scores', {}).items():
                if score is not None:
                    click.echo(f"  {metric.upper()}: {score:.3f}")
                else:
                    click.echo(f"  {metric.upper()}: N/A (skipped)")
            click.echo(f"\nResults saved to: {pipeline.results_dir}")

    # Generate overall summary for all languages
    if execution_id and all_results:
        _generate_overall_summary(all_results, base_results_dir, manifest_path)


def _evaluate_ground_truth_mode(types_to_eval, pipeline, data_dir, samples, limit, metrics, all_results):
    """Evaluate in ground truth mode (by translation type)."""
    for trans_type in types_to_eval:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating type: {trans_type or 'auto-detect'}")
        logger.info(f"{'='*60}")

        # Load samples
        sample_list = list(samples) if samples else None
        loaded_samples, errors = load_samples(
            data_dir=data_dir,
            translation_type=trans_type,
            sample_uuids=sample_list,
        )

        if not loaded_samples:
            logger.warning(f"No samples loaded for type: {trans_type}")
            continue

        # Apply limit if specified
        if limit:
            logger.info(f"Limiting evaluation to {limit} samples")
            loaded_samples = loaded_samples[:limit]

        # Evaluate
        results = pipeline.evaluate_samples(
            loaded_samples,
            metrics=list(metrics) if metrics else None,
        )

        if results:
            # Save results
            pipeline.save_results(results)

            # Generate visualizations
            pipeline.generate_visualizations(results)

            all_results.append(results)

            # Print summary
            click.echo(f"\n{'='*60}")
            click.echo(f"EVALUATION COMPLETE - {trans_type}")
            click.echo(f"{'='*60}")
            click.echo(f"Run ID: {pipeline.run_id}")
            click.echo(f"Samples evaluated: {results['valid_samples']}/{results['total_samples']}")
            click.echo(f"\nAggregate Scores:")
            for metric, score in results.get('aggregate_scores', {}).items():
                if score is not None:
                    click.echo(f"  {metric.upper()}: {score:.3f}")
                else:
                    click.echo(f"  {metric.upper()}: N/A (skipped)")
            click.echo(f"\nResults saved to: {pipeline.results_dir}")


def _generate_overall_summary(all_results, base_results_dir, manifest_path):
    """Generate overall summary across all languages."""
    logger.info("Generating overall summary across all languages...")

    # Read the updated manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Compute average scores across languages
    all_metrics = set()
    for lang_results in all_results:
        all_metrics.update(lang_results.get('aggregate_scores', {}).keys())

    import statistics

    overall_scores = {}
    for metric in all_metrics:
        scores = [
            r['aggregate_scores'].get(metric)
            for r in all_results
            if r.get('aggregate_scores', {}).get(metric) is not None
        ]
        if scores:
            overall_scores[metric] = {
                'mean': sum(scores) / len(scores),
                'median': statistics.median(scores) if len(scores) > 0 else 0,
                'std': statistics.stdev(scores) if len(scores) > 1 else 0,
                'min': min(scores),
                'max': max(scores),
            }

    # Add overall summary to manifest
    manifest['overall_summary'] = {
        'languages_evaluated': len(all_results),
        'average_scores': overall_scores,
    }

    # Save summary file
    summary_path = base_results_dir / 'overall_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved overall summary: {summary_path}")

    # Generate cross-language visualizations
    from scripts.visualizations import create_language_metric_heatmap

    viz_dir = base_results_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    logger.info(f"Generating cross-language visualizations in {viz_dir}")

    # Language × Metric Performance Heatmap
    try:
        create_language_metric_heatmap(manifest, viz_dir / 'language_metric_heatmap.png')
    except Exception as e:
        logger.error(f"Failed to create language×metric heatmap: {e}", exc_info=True)

    # Update manifest with overall summary
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Print overall summary
    click.echo(f"\n{'='*70}")
    click.echo("OVERALL SUMMARY - ALL LANGUAGES")
    click.echo(f"{'='*70}")
    click.echo(f"Execution ID: {manifest['execution_id']}")
    click.echo(f"Languages evaluated: {len(all_results)}")
    click.echo(f"Total samples: {manifest['total_samples']}")
    click.echo(f"Total valid samples: {manifest['total_valid_samples']}")
    click.echo(f"\nAggregate Statistics Across Languages:")
    for metric, stats in overall_scores.items():
        click.echo(f"  {metric.upper()}:")
        click.echo(f"    Mean:   {stats['mean']:.3f}")
        click.echo(f"    Median: {stats['median']:.3f}")
        click.echo(f"    Std:    {stats['std']:.3f}")
        click.echo(f"    Range:  {stats['min']:.3f} - {stats['max']:.3f}")
    click.echo(f"\nPer-Language Breakdown:")
    for lang, lang_data in manifest['languages'].items():
        click.echo(f"  {lang.upper()}:")
        click.echo(f"    Samples: {lang_data['valid_samples']}/{lang_data['total_samples']}")
        for metric, score in lang_data['aggregate_scores'].items():
            click.echo(f"    {metric.upper()}: {score:.3f}")
    click.echo(f"\nResults directory: {base_results_dir}")
    click.echo(f"Overall summary: {summary_path}")
    click.echo(f"\nCross-language visualizations:")
    click.echo(f"  - Language×Metric Heatmap: {viz_dir / 'language_metric_heatmap.png'}")


if __name__ == '__main__':
    main()
