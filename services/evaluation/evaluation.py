"""
Main evaluation script for multimodal translation evaluation.

Evaluates translations across different modalities using appropriate metrics.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import click
import pandas as pd
import yaml
from tqdm import tqdm

# Import our modules
from data_loader import load_samples, validate_sample_for_metrics, TranslationSample
from text_metrics import TextMetrics
from audio_metrics import AudioMetrics
from scripts.comet_evaluator import CometEvaluator
from scripts.blaser_evaluator import BlaserEvaluator
from visualizations import generate_all_visualizations, generate_html_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Metric selection logic based on translation type
METRIC_MAPPING = {
    'text_to_text': ['bleu', 'chrf', 'comet'],
    'audio_to_text': ['bleu', 'chrf', 'comet'],
    'text_to_audio': ['bleu', 'chrf', 'comet', 'mcd'],
    'audio_to_audio': ['bleu', 'chrf', 'comet', 'mcd', 'blaser'],
}


class EvaluationPipeline:
    """Main evaluation pipeline orchestrator."""
    
    def __init__(
        self,
        output_dir: Path,
        run_id: Optional[str] = None,
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            output_dir: Directory to save results
            run_id: Run identifier (auto-generated if None)
        """
        self.output_dir = Path(output_dir)
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create output directories
        self.results_dir = self.output_dir / self.run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = self.results_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
        self.viz_dir = self.results_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
        # Setup file logging
        file_handler = logging.FileHandler(self.logs_dir / 'evaluation.log')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)
        
        # Initialize metric evaluators (lazy loading)
        self.text_metrics = None
        self.audio_metrics = None
        self.comet_evaluator = None
        self.blaser_evaluator = None
    
    def _get_metrics_for_type(self, translation_type: str) -> List[str]:
        """Get appropriate metrics for translation type."""
        return METRIC_MAPPING.get(translation_type, ['bleu', 'chrf'])
    
    def _load_evaluators(self, metrics: List[str]):
        """Load required evaluators based on metrics."""
        if any(m in metrics for m in ['bleu', 'chrf']):
            if self.text_metrics is None:
                self.text_metrics = TextMetrics()
                logger.info("Initialized text metrics evaluator")
        
        if 'comet' in metrics:
            if self.comet_evaluator is None:
                self.comet_evaluator = CometEvaluator()
                logger.info("Initialized COMET evaluator")
        
        if 'mcd' in metrics:
            if self.audio_metrics is None:
                self.audio_metrics = AudioMetrics()
                logger.info("Initialized audio metrics evaluator")
        
        if 'blaser' in metrics:
            if self.blaser_evaluator is None:
                self.blaser_evaluator = BlaserEvaluator()
                logger.info("Initialized BLASER evaluator")
    
    def evaluate_samples(
        self,
        samples: List[TranslationSample],
        metrics: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluate a list of samples.
        
        Args:
            samples: List of TranslationSample objects
            metrics: List of metrics to compute (auto-detect if None)
            
        Returns:
            Dictionary with evaluation results
        """
        if not samples:
            logger.error("No samples to evaluate")
            return {}
        
        # Auto-detect translation type and metrics
        translation_type = samples[0].translation_type
        if metrics is None:
            metrics = self._get_metrics_for_type(translation_type)
        
        logger.info(f"Evaluating {len(samples)} samples of type '{translation_type}'")
        logger.info(f"Metrics to compute: {metrics}")
        
        # Load evaluators
        self._load_evaluators(metrics)
        
        # Validate samples
        valid_samples = []
        skipped_samples = []
        
        for sample in samples:
            is_valid, missing = validate_sample_for_metrics(sample, metrics)
            if is_valid:
                valid_samples.append(sample)
            else:
                skipped_samples.append({
                    'uuid': sample.uuid,
                    'missing': missing
                })
                logger.warning(f"Skipping sample {sample.uuid}: missing {missing}")
        
        logger.info(f"Valid samples: {len(valid_samples)}, Skipped: {len(skipped_samples)}")
        
        if not valid_samples:
            logger.error("No valid samples to evaluate")
            return {}
        
        # Prepare data for metrics
        results = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'translation_type': translation_type,
            'language_pair': f"{valid_samples[0].source_language}-{valid_samples[0].target_language}",
            'total_samples': len(samples),
            'valid_samples': len(valid_samples),
            'skipped_samples': len(skipped_samples),
            'metrics_computed': metrics,
            'aggregate_scores': {},
            'score_statistics': {},
            'per_sample_results': [],
        }
        
        # Extract texts and paths
        sources = []
        hypotheses = []
        references = []
        source_audio_paths = []
        target_audio_paths = []
        
        for sample in valid_samples:
            # For text metrics, use source or transcribed text
            if translation_type in ['text_to_text', 'text_to_audio']:
                src = sample.source_text
            else:  # audio_to_text, audio_to_audio
                src = sample.transcribed_text or sample.source_text or ""
            
            sources.append(src)
            hypotheses.append(sample.target_text or "")
            references.append([sample.target_text or ""])  # Single reference
            
            if sample.source_audio_path:
                source_audio_paths.append(sample.source_audio_path)
            if sample.target_audio_path:
                target_audio_paths.append(sample.target_audio_path)
        
        # Compute text metrics
        if 'bleu' in metrics:
            logger.info("Computing BLEU scores...")
            bleu_result = self.text_metrics.compute_bleu(hypotheses, references)
            results['aggregate_scores']['bleu'] = bleu_result['corpus_score']
            results['bleu_details'] = bleu_result
            
        if 'chrf' in metrics:
            logger.info("Computing chrF scores...")
            chrf_result = self.text_metrics.compute_chrf(hypotheses, references)
            results['aggregate_scores']['chrf'] = chrf_result['corpus_score']
            results['chrf_details'] = chrf_result
        
        # Compute COMET
        if 'comet' in metrics:
            logger.info("Computing COMET scores...")
            try:
                # Convert references from [[ref]] to [ref]
                refs_flat = [r[0] if isinstance(r, list) and len(r) > 0 else r for r in references]
                comet_result = self.comet_evaluator.evaluate(sources, hypotheses, refs_flat)
                results['aggregate_scores']['comet'] = comet_result['corpus_score']
                results['comet_details'] = comet_result
            except Exception as e:
                logger.error(f"COMET evaluation failed: {e}")
                results['aggregate_scores']['comet'] = 0.0
        
        # Compute MCD
        if 'mcd' in metrics and target_audio_paths:
            logger.info("Computing MCD scores...")
            # TODO: MCD requires separate reference audio files for proper evaluation.
            # Currently skipping MCD as comparing generated audio to itself (target_audio_paths)
            # produces meaningless results (MCD=0). Need to update data schema to include
            # reference_audio.wav files for each sample.
            logger.warning("MCD computation skipped: requires separate reference audio files")
            logger.warning("Data schema needs reference_audio.wav in addition to target_audio.wav")
            results['aggregate_scores']['mcd'] = None
            results['mcd_details'] = {
                'error': 'MCD requires reference audio - data schema update needed',
                'mean_mcd': None,
                'mcd_scores': [],
            }

            # Uncomment when reference audio is available:
            # try:
            #     mcd_result = self.audio_metrics.compute_mcd_batch(
            #         target_audio_paths,       # Generated audio
            #         reference_audio_paths,    # Reference audio (needs to be added)
            #     )
            #     results['aggregate_scores']['mcd'] = mcd_result['mean_mcd']
            #     results['mcd_details'] = mcd_result
            # except Exception as e:
            #     logger.error(f"MCD evaluation failed: {e}")
            #     results['aggregate_scores']['mcd'] = None
        
        # Compute BLASER
        if 'blaser' in metrics and source_audio_paths and target_audio_paths:
            logger.info("Computing BLASER scores...")
            try:
                refs_flat = [r[0] if isinstance(r, list) and len(r) > 0 else r for r in references]

                # Get language codes, use defaults if not specified
                source_lang_code = valid_samples[0].source_language or 'eng'
                target_lang_code = valid_samples[0].target_language or 'spa'

                blaser_result = self.blaser_evaluator.evaluate(
                    source_audio_paths,
                    target_audio_paths,
                    refs_flat,
                    sources,
                    source_lang=f"{source_lang_code}_Latn",
                    target_lang=f"{target_lang_code}_Latn",
                )
                results['aggregate_scores']['blaser'] = blaser_result['corpus_score']
                results['blaser_details'] = blaser_result
            except Exception as e:
                logger.error(f"BLASER evaluation failed: {e}")
                results['aggregate_scores']['blaser'] = 0.0
        
        # Compute score statistics
        import numpy as np
        for metric in metrics:
            if metric in ['bleu', 'chrf', 'comet', 'blaser']:
                details_key = f'{metric}_details'
                if details_key in results and 'sentence_scores' in results[details_key]:
                    scores = results[details_key]['sentence_scores']
                    if scores:
                        results['score_statistics'][metric] = {
                            'mean': float(np.mean(scores)),
                            'std': float(np.std(scores)),
                            'min': float(np.min(scores)),
                            'max': float(np.max(scores)),
                        }
            elif metric == 'mcd' and 'mcd_details' in results:
                mcd_scores = results['mcd_details'].get('mcd_scores', [])
                if mcd_scores:
                    results['score_statistics']['mcd'] = {
                        'mean': float(np.mean(mcd_scores)),
                        'std': float(np.std(mcd_scores)),
                        'min': float(np.min(mcd_scores)),
                        'max': float(np.max(mcd_scores)),
                    }
        
        # Prepare per-sample results
        for i, sample in enumerate(valid_samples):
            sample_result = {
                'uuid': sample.uuid,
                'source_language': sample.source_language,
                'target_language': sample.target_language,
            }
            
            for metric in metrics:
                if metric in ['bleu', 'chrf', 'comet', 'blaser']:
                    details_key = f'{metric}_details'
                    if details_key in results and 'sentence_scores' in results[details_key]:
                        scores = results[details_key]['sentence_scores']
                        if i < len(scores):
                            sample_result[f'{metric}_score'] = scores[i]
                elif metric == 'mcd' and 'mcd_details' in results:
                    mcd_scores = results['mcd_details'].get('mcd_scores', [])
                    if i < len(mcd_scores):
                        sample_result['mcd_score'] = mcd_scores[i]
            
            results['per_sample_results'].append(sample_result)
        
        return results
    
    def save_results(self, results: Dict):
        """Save evaluation results to files."""
        # Save summary JSON
        summary_path = self.results_dir / 'summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            # Remove large details for summary
            summary = {k: v for k, v in results.items() 
                      if not k.endswith('_details')}
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved summary to {summary_path}")
        
        # Save detailed CSV
        if results.get('per_sample_results'):
            df = pd.DataFrame(results['per_sample_results'])
            csv_path = self.results_dir / 'detailed_results.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved detailed results to {csv_path}")
        
        # Save full results JSON
        full_path = self.results_dir / 'per_sample_results.json'
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved full results to {full_path}")
    
    def generate_visualizations(self, results: Dict):
        """Generate all visualizations."""
        logger.info("Generating visualizations...")

        if results.get('per_sample_results'):
            df = pd.DataFrame(results['per_sample_results'])

            # Auto-detect primary metric based on translation type
            translation_type = results.get('translation_type')
            metrics_computed = results.get('metrics_computed', [])

            if translation_type == 'audio_to_audio' and 'blaser' in metrics_computed:
                primary_metric = 'blaser'
            elif 'comet' in metrics_computed:
                primary_metric = 'comet'
            elif 'bleu' in metrics_computed:
                primary_metric = 'bleu'
            else:
                primary_metric = None

            logger.info(f"Using primary metric for quality categorization: {primary_metric}")

            generate_all_visualizations(results, df, self.viz_dir, primary_metric=primary_metric)

            # Generate HTML report
            html_path = self.results_dir / 'summary_report.html'
            generate_html_report(results, self.viz_dir, html_path)
            logger.info(f"Generated HTML report at {html_path}")


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
def main(
    translation_type: Optional[str],
    data_dir: str,
    output_dir: str,
    samples: tuple,
    config: Optional[str],
    run_id: Optional[str],
    metrics: tuple,
):
    """
    Evaluate multimodal translation system.
    
    Examples:
    
        \b
        # Evaluate text-to-text translations
        python evaluation.py --type text_to_text --data-dir ../data/app_evaluation/text_to_text
        
        \b
        # Evaluate all translation types
        python evaluation.py --type all --data-dir ../data/app_evaluation
        
        \b
        # Evaluate specific samples
        python evaluation.py --type text_to_audio --samples uuid1 uuid2 --data-dir ../data/app_evaluation/text_to_audio
        
        \b
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
        
        # Initialize pipeline
        pipeline = EvaluationPipeline(output_dir=output_dir, run_id=run_id)
        
        logger.info(f"Starting evaluation with run_id: {pipeline.run_id}")
        
        # Determine translation types to evaluate
        if translation_type == 'all':
            types_to_eval = ['text_to_text', 'audio_to_text', 'text_to_audio', 'audio_to_audio']
        else:
            types_to_eval = [translation_type] if translation_type else [None]
        
        # Evaluate each type
        all_results = []
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
                    click.echo(f"  {metric.upper()}: {score:.3f}")
                click.echo(f"\nResults saved to: {pipeline.results_dir}")
        
        if not all_results:
            logger.error("No results generated")
            sys.exit(1)
        
        click.echo(f"\n{'='*60}")
        click.echo("ALL EVALUATIONS COMPLETE")
        click.echo(f"{'='*60}")
        click.echo(f"Results directory: {pipeline.results_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
