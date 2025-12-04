"""
Evaluation pipeline orchestrator.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.data_loader import TranslationSample, validate_sample_for_metrics
from scripts.text_metrics import TextMetrics
from scripts.audio_metrics import AudioMetrics
from scripts.comet_evaluator import CometEvaluator
from scripts.blaser_evaluator import BlaserEvaluator
from scripts.visualizations import generate_all_visualizations, generate_html_report

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Main evaluation pipeline orchestrator."""

    def __init__(
        self,
        output_dir: Path,
        run_id: Optional[str] = None,
        use_run_id_subdir: bool = True,
    ):
        """
        Initialize evaluation pipeline.

        Args:
            output_dir: Directory to save results
            run_id: Run identifier (auto-generated if None)
            use_run_id_subdir: If True, creates output_dir/run_id/ structure (default).
                               If False, uses output_dir directly (for custom directory structures).
        """
        self.output_dir = Path(output_dir)
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create output directories
        if use_run_id_subdir:
            # Traditional mode: results/run_id/
            self.results_dir = self.output_dir / self.run_id
        else:
            # Custom mode: use output_dir directly (e.g., results/execution_id/language/)
            self.results_dir = self.output_dir

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
            # Default to all available metrics
            metrics = ['bleu', 'chrf', 'comet', 'mcd', 'blaser']

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
        predicted_audio_paths = []

        # Check if we're evaluating predictions (samples have predicted_tgt_text populated)
        has_predictions = any(sample.predicted_tgt_text for sample in valid_samples)

        for sample in valid_samples:
            # For text metrics, use source or transcribed text
            if translation_type in ['text_to_text', 'text_to_audio']:
                src = sample.source_text
            else:  # audio_to_text, audio_to_audio
                src = sample.transcribed_text or sample.source_text or ""

            sources.append(src)

            # For predictions mode: use predicted_tgt_text as hypothesis, target_text as reference
            # For ground truth mode: use target_text as both (measuring reference quality)
            if has_predictions:
                hypotheses.append(sample.predicted_tgt_text or "")
                references.append([sample.target_text or ""])  # Ground truth as reference
            else:
                hypotheses.append(sample.target_text or "")
                references.append([sample.target_text or ""])  # Single reference

            if sample.source_audio_path:
                source_audio_paths.append(sample.source_audio_path)
            if sample.target_audio_path:
                target_audio_paths.append(sample.target_audio_path)
            if sample.predicted_tgt_audio_path:
                predicted_audio_paths.append(sample.predicted_tgt_audio_path)

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
        if 'mcd' in metrics:
            # For predictions mode: compare predicted audio against ground truth
            # For ground truth mode: skip (would compare audio to itself)
            if has_predictions and predicted_audio_paths and target_audio_paths:
                logger.info("Computing MCD scores...")
                logger.info("Comparing predicted audio against ground truth audio")
                try:
                    mcd_result = self.audio_metrics.compute_mcd_batch(
                        predicted_audio_paths,  # Hypothesis (TTS-generated audio)
                        target_audio_paths,     # Reference (ground truth audio)
                    )
                    results['aggregate_scores']['mcd'] = mcd_result['mean_mcd']
                    results['mcd_details'] = mcd_result
                except Exception as e:
                    logger.error(f"MCD evaluation failed: {e}")
                    results['aggregate_scores']['mcd'] = None
            else:
                # Ground truth mode or missing audio files
                logger.warning("MCD computation skipped: requires both predicted and ground truth audio")
                if not has_predictions:
                    logger.warning("MCD only available in predictions mode")
                results['aggregate_scores']['mcd'] = None
                results['mcd_details'] = {
                    'error': 'MCD requires predictions mode with both predicted and ground truth audio',
                    'mean_mcd': None,
                    'mcd_scores': [],
                }

        # Compute BLASER
        # Use predicted audio if available, otherwise use target audio
        blaser_target_audio = predicted_audio_paths if predicted_audio_paths else target_audio_paths

        if 'blaser' in metrics and source_audio_paths and blaser_target_audio:
            logger.info("Computing BLASER scores...")
            if predicted_audio_paths:
                logger.info("Using predicted audio for BLASER evaluation")
            try:
                refs_flat = [r[0] if isinstance(r, list) and len(r) > 0 else r for r in references]

                # Get language codes, use defaults if not specified
                source_lang_code = valid_samples[0].source_language or 'eng'
                target_lang_code = valid_samples[0].target_language or 'spa'

                blaser_result = self.blaser_evaluator.evaluate(
                    source_audio_paths,
                    blaser_target_audio,
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
        import json

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
