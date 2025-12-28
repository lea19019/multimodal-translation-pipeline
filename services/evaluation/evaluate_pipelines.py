#!/usr/bin/env python3
"""
Simplified Pipeline Evaluation - CSV Output Only

Evaluates a single pipeline across languages and generates:
- {lang}_sample_scores.csv: Per-sample metrics
- {lang}_scores.csv: Aggregate statistics per language
- all_scores.csv: Overall statistics across languages

Usage:
    uv run evaluate_pipelines.py --pipeline pipeline_1
    uv run evaluate_pipelines.py --pipeline pipeline_1 --execution-id 20251214_143022
    uv run evaluate_pipelines.py --pipeline pipeline_9 --languages efik
    uv run evaluate_pipelines.py --pipeline pipeline_1 --limit 10
"""

# CRITICAL: Set offline mode BEFORE any imports to prevent HuggingFace downloads
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import logging
import json
import sys
import click
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent / 'config'))
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from pipeline_config import get_pipeline, get_synthesis_path, LANGUAGES
from text_metrics import TextMetrics
from comet_evaluator import CometEvaluator
from audio_metrics import AudioMetrics
from blaser_evaluator import BlaserEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Simplified evaluator for a single pipeline - CSV output only."""

    def __init__(self, pipeline_id: str, execution_id: Optional[str] = None, output_dir: Path = None):
        """
        Initialize evaluator.

        Args:
            pipeline_id: Pipeline identifier (e.g., 'pipeline_1')
            execution_id: Execution identifier (auto-generated if None)
            output_dir: Base output directory (default: services/evaluation/results)
        """
        self.pipeline_id = pipeline_id
        self.pipeline = get_pipeline(pipeline_id)

        if self.pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        if execution_id is None:
            execution_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.execution_id = execution_id

        # Output: results/{pipeline_id}_{short_name}_{execution_id}/
        if output_dir is None:
            output_dir = Path(__file__).parent / 'results'
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / f"{pipeline_id}_{self.pipeline['short_name']}_{execution_id}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized evaluator for {pipeline_id}")
        logger.info(f"Results directory: {self.results_dir}")

        # Initialize manifest
        self.manifest_path = self.results_dir / 'manifest.json'
        self.manifest = self._load_or_create_manifest()

        # Initialize metric evaluators (lazy loading)
        self.text_metrics = None
        self.comet = None
        self.audio_metrics = None
        self.blaser = None

    def _load_or_create_manifest(self) -> Dict:
        """Load existing manifest or create new one."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            logger.info(f"Loaded existing manifest: {self.manifest_path}")
            return manifest
        else:
            manifest = {
                'execution_id': self.execution_id,
                'pipeline_id': self.pipeline_id,
                'pipeline_name': self.pipeline['name'],
                'pipeline_short_name': self.pipeline['short_name'],
                'started_at': datetime.now().isoformat(),
                'completed_at': None,
                'languages': {}
            }
            self._save_manifest(manifest)
            return manifest

    def _save_manifest(self, manifest: Dict = None):
        """Save manifest to disk."""
        if manifest is None:
            manifest = self.manifest
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _update_language_status(self, language: str, status: str, **kwargs):
        """Update language status in manifest."""
        if language not in self.manifest['languages']:
            self.manifest['languages'][language] = {}
        
        self.manifest['languages'][language]['status'] = status
        self.manifest['languages'][language].update(kwargs)
        self._save_manifest()

    def _initialize_metrics(self):
        """Initialize metric evaluators (lazy loading)."""
        if self.text_metrics is None and self.pipeline['uses_nmt']:
            self.text_metrics = TextMetrics()
            logger.info("Initialized text metrics evaluator")

        if self.comet is None and 'comet' in self.pipeline['metrics']:
            self.comet = CometEvaluator()
            logger.info("Initialized COMET evaluator")

        if self.audio_metrics is None and 'mcd' in self.pipeline['metrics']:
            self.audio_metrics = AudioMetrics()
            logger.info("Initialized audio metrics evaluator")

        if self.blaser is None and 'blaser' in self.pipeline['metrics']:
            self.blaser = BlaserEvaluator()
            logger.info("Initialized BLASER evaluator")

    def evaluate_language(self, language: str, limit: Optional[int] = None) -> Dict:
        """
        Evaluate pipeline for a single language.

        Args:
            language: Language name (e.g., 'efik')
            limit: Limit number of samples (for testing)

        Returns:
            Dict with aggregate statistics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluating: {self.pipeline['name']} × {language}")
        logger.info(f"{'='*70}")

        self._update_language_status(language, 'in_progress')

        try:
            # Initialize metrics
            self._initialize_metrics()

            # Get data paths
            paths = get_synthesis_path(self.pipeline_id, language)

            logger.info(f"Loading synthesis results: {paths['csv_path'].name}")

            # Load predicted samples
            if not paths['csv_path'].exists():
                raise FileNotFoundError(f"Predicted CSV not found: {paths['csv_path']}")

            # Read CSV - try pipe delimiter first (newer format), then comma
            try:
                pred_df = pd.read_csv(paths['csv_path'], delimiter='|')
                logger.info(f"  ✓ Loaded {len(pred_df)} samples")
            except Exception as e1:
                try:
                    pred_df = pd.read_csv(paths['csv_path'])
                    logger.info(f"  ✓ Loaded {len(pred_df)} samples")
                except Exception as e2:
                    logger.error(f"Failed to parse CSV with pipe or comma delimiter")
                    raise e2

            # Filter successful syntheses only
            successful_df = pred_df[pred_df['success'] == True].copy()
            logger.info(f"  ✓ Successful syntheses: {len(successful_df)}/{len(pred_df)}")

            # Remove duplicates (if any)
            if successful_df['segment_id'].duplicated().any():
                dup_count = successful_df['segment_id'].duplicated().sum()
                logger.warning(f"  ⚠️  Found {dup_count} duplicate segment_ids, removing...")
                successful_df = successful_df.drop_duplicates(subset='segment_id', keep='first')

            logger.info(f"  ✓ After deduplication: {len(successful_df)} samples")

            # Load ground truth data for text metrics
            if self.pipeline['uses_nmt']:
                logger.info(f"\nLoading ground truth references: {paths['nmt_csv_path'].name}")
                if not paths['nmt_csv_path'].exists():
                    raise FileNotFoundError(f"NMT CSV not found: {paths['nmt_csv_path']}")

                # Try pipe delimiter first, then comma
                try:
                    nmt_df = pd.read_csv(paths['nmt_csv_path'], delimiter='|')
                except Exception:
                    nmt_df = pd.read_csv(paths['nmt_csv_path'])

                # Deduplicate NMT CSV (it may have multiple rows per segment_id)
                nmt_original_count = len(nmt_df)
                nmt_df = nmt_df.drop_duplicates(subset='segment_id', keep='first')
                if nmt_original_count != len(nmt_df):
                    logger.info(f"  ✓ Deduplicated NMT CSV: {nmt_original_count} → {len(nmt_df)} rows")

                # Simple merge to get only ground truth columns
                df = successful_df.merge(
                    nmt_df[['segment_id', 'ground_truth_tgt_text', 'src_text']],
                    on='segment_id',
                    how='left'
                )
                logger.info(f"  ✓ Merged {len(df)} samples with ground truth")

                # Log what we're comparing
                logger.info(f"\nText metrics will compare:")
                logger.info(f"  - Hypothesis: 'text' column from synthesis CSV (input to TTS)")
                logger.info(f"  - Reference: 'ground_truth_tgt_text' from NMT CSV")
            else:
                df = successful_df

            # Apply limit if specified
            if limit is not None:
                df = df.head(limit)
                logger.info(f"\n⚙️  Limited to {limit} samples for testing")

            logger.info(f"\nEvaluating {len(df)} samples...")

            # Compute metrics
            results = self._compute_metrics(df, paths, language)

            # Save sample scores
            sample_scores_path = self.results_dir / f"{language}_sample_scores.csv"
            results['df'].to_csv(sample_scores_path, index=False)
            logger.info(f"💾 Saved sample scores: {sample_scores_path}")

            # Compute and save aggregate statistics
            agg_stats = self._compute_aggregate_stats(results['df'])
            agg_stats_path = self.results_dir / f"{language}_scores.csv"
            agg_stats.to_csv(agg_stats_path, index=False)
            logger.info(f"💾 Saved aggregate statistics: {agg_stats_path}")

            # Update manifest
            self._update_language_status(
                language, 
                'completed',
                sample_count=len(df),
                metrics=results['summary']
            )

            logger.info(f"✓ Completed: {self.pipeline['name']} × {language}")
            return results['summary']

        except Exception as e:
            logger.error(f"Failed to evaluate {language}: {e}", exc_info=True)
            self._update_language_status(language, 'failed', error=str(e))
            raise

    def _compute_metrics(self, df: pd.DataFrame, paths: Dict, language: str) -> Dict:
        """Compute all metrics for the samples."""
        results_df = df.copy()
        summary = {}

        # Text metrics (if pipeline uses NMT)
        if self.pipeline['uses_nmt']:
            logger.info("\n" + "="*70)
            logger.info("TEXT METRICS")
            logger.info("="*70)

            # Extract text columns
            # Predicted CSV has: text (TTS output), segment_id
            # NMT CSV has: src_text, predicted_tgt_text, ground_truth_tgt_text, segment_id
            sources = df['src_text'].tolist()
            predictions = df['text'].tolist()  # TTS pipeline prediction
            references_flat = df['ground_truth_tgt_text'].tolist()  # Flat list for COMET
            references_nested = [[ref] for ref in references_flat]  # Nested list for BLEU/chrF

            # BLEU and chrF
            logger.info("\n📊 Computing BLEU...")
            bleu_result = self.text_metrics.compute_bleu(predictions, references_nested)
            results_df['bleu'] = bleu_result['sentence_scores']
            summary['bleu'] = bleu_result['corpus_score']
            logger.info(f"  Corpus BLEU: {summary['bleu']:.2f}")

            logger.info("\n📊 Computing chrF++...")
            chrf_result = self.text_metrics.compute_chrf(predictions, references_nested)
            results_df['chrf'] = chrf_result['sentence_scores']
            summary['chrf'] = chrf_result['corpus_score']
            logger.info(f"  Corpus chrF++: {summary['chrf']:.2f}")

            # COMET
            if self.comet is not None:
                logger.info("\n📊 Computing COMET (SSA-COMET-QE for African languages)...")
                comet_result = self.comet.evaluate(sources, predictions, references_flat)
                results_df['comet'] = comet_result['sentence_scores']
                summary['comet'] = comet_result['corpus_score']
                logger.info(f"  Corpus COMET: {summary['comet']:.4f}")

        # Audio metrics
        logger.info("\n" + "="*70)
        logger.info("AUDIO METRICS")
        logger.info("="*70)

        # Validate audio files before computing metrics
        audio_pairs = []
        missing_audio = []

        for _, row in df.iterrows():
            segment_id = row['segment_id']
            audio_filename = row.get('audio_filename', '')

            if not audio_filename:
                missing_audio.append(f"segment {segment_id}: no audio_filename in CSV")
                continue

            # Predicted audio uses the filename from CSV
            pred_audio = paths['audio_dir'] / audio_filename

            # Reference audio: replace _pred.wav with .wav
            ref_filename = audio_filename.replace('_pred.wav', '.wav')
            ref_audio = paths['ref_audio_dir'] / ref_filename

            # Check file existence
            pred_exists = pred_audio.exists()
            ref_exists = ref_audio.exists()

            if not pred_exists:
                missing_audio.append(f"segment {segment_id}: predicted audio missing ({audio_filename})")
            if not ref_exists:
                missing_audio.append(f"segment {segment_id}: reference audio missing ({ref_filename})")

            if pred_exists and ref_exists:
                audio_pairs.append({
                    'segment_id': segment_id,
                    'predicted': str(pred_audio),
                    'reference': str(ref_audio),
                    'src_text': row.get('src_text', ''),
                    'ref_text': row.get('ground_truth_tgt_text', '')
                })

        # Report audio validation results
        logger.info(f"\nAudio validation:")
        logger.info(f"  ✓ Found {len(audio_pairs)}/{len(df)} valid audio pairs")

        if missing_audio:
            logger.warning(f"  ⚠️  Missing audio for {len(missing_audio)} samples:")
            for msg in missing_audio[:5]:  # Show first 5
                logger.warning(f"    - {msg}")
            if len(missing_audio) > 5:
                logger.warning(f"    ... and {len(missing_audio) - 5} more")

        # Check if audio metrics are required
        audio_metrics_required = ('mcd' in self.pipeline['metrics'] or 'blaser' in self.pipeline['metrics'])

        if audio_metrics_required and len(audio_pairs) == 0:
            raise RuntimeError(
                f"Audio metrics are required for this pipeline but no audio files were found.\n"
                f"Expected audio directory: {paths['audio_dir']}\n"
                f"Expected reference directory: {paths['ref_audio_dir']}"
            )

        if len(audio_pairs) > 0:
            # MCD
            if self.audio_metrics is not None:
                logger.info("\n🎵 Computing MCD (Mel-Cepstral Distance)...")
                mcd_batch_result = self.audio_metrics.compute_mcd_batch(
                    [p['predicted'] for p in audio_pairs],
                    [p['reference'] for p in audio_pairs]
                )
                
                # Map back to dataframe
                mcd_map = {p['segment_id']: mcd for p, mcd in zip(audio_pairs, mcd_batch_result['mcd_scores'])}
                results_df['mcd'] = results_df['segment_id'].map(mcd_map)
                
                summary['mcd'] = mcd_batch_result['mean_mcd']
                logger.info(f"  Mean MCD: {summary['mcd']:.2f} (lower is better)")

            # BLASER
            if self.blaser is not None:
                logger.info("\n🎵 Computing BLASER 2.0 (Speech-to-Speech Quality)...")
                
                # Get language code for BLASER (e.g., 'efi' -> need FLORES code)
                iso_code = LANGUAGES.get(language, 'efi')
                target_lang_code = f"{iso_code}_Latn"  # FLORES format
                
                # BLASER needs source audio (reference), target audio (predicted), texts, and language codes
                blaser_result = self.blaser.evaluate(
                    source_audio_paths=[p['reference'] for p in audio_pairs],
                    target_audio_paths=[p['predicted'] for p in audio_pairs],
                    source_texts=[p['src_text'] for p in audio_pairs],
                    reference_texts=[p['ref_text'] for p in audio_pairs],
                    source_lang='eng_Latn',
                    target_lang=target_lang_code
                )
                
                # Map back to dataframe
                blaser_map = {p['segment_id']: score for p, score in zip(audio_pairs, blaser_result['sentence_scores'])}
                results_df['blaser'] = results_df['segment_id'].map(blaser_map)
                summary['blaser'] = blaser_result['corpus_score']
                logger.info(f"  Corpus BLASER: {summary['blaser']:.4f} (0-5, higher is better)")

        return {
            'df': results_df,
            'summary': summary
        }

    def _compute_aggregate_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute aggregate statistics for each metric."""
        stats_rows = []
        
        for metric in ['bleu', 'chrf', 'comet', 'mcd', 'blaser']:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    stats_rows.append({
                        'metric': metric,
                        'mean': values.mean(),
                        'median': values.median(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'count': len(values)
                    })
        
        return pd.DataFrame(stats_rows)

    def evaluate_all(self, languages: Optional[List[str]] = None, limit: Optional[int] = None):
        """
        Evaluate pipeline for all supported languages.

        Args:
            languages: List of languages to evaluate (default: all supported by pipeline)
            limit: Limit number of samples (for testing)
        """
        # Get supported languages for this pipeline
        if languages is None:
            languages = self.pipeline.get('languages', list(LANGUAGES.keys()))

        logger.info(f"Starting evaluation of {len(languages)} languages")
        
        all_results = {}
        
        for lang in languages:
            # Check if already completed
            if lang in self.manifest['languages'] and self.manifest['languages'][lang].get('status') == 'completed':
                logger.info(f"Skipping {lang} (already completed)")
                all_results[lang] = self.manifest['languages'][lang].get('metrics', {})
                continue
            
            try:
                results = self.evaluate_language(lang, limit)
                all_results[lang] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {lang}: {e}")
                continue

        # Generate all_scores.csv
        if all_results:
            self._generate_all_scores_csv(all_results)

        # Mark as completed
        self.manifest['completed_at'] = datetime.now().isoformat()
        self._save_manifest()

        logger.info(f"\n{'='*70}")
        logger.info("EVALUATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Languages evaluated: {len(all_results)}")

    def _generate_all_scores_csv(self, all_results: Dict):
        """Generate all_scores.csv with aggregate statistics across languages."""
        rows = []
        
        for lang, metrics in all_results.items():
            row = {'language': lang}
            
            for metric in ['bleu', 'chrf', 'comet', 'mcd', 'blaser']:
                if metric in metrics:
                    row[f'{metric}_mean'] = metrics[metric]
                    
                    # Also load individual language stats for std
                    lang_stats_path = self.results_dir / f"{lang}_scores.csv"
                    if lang_stats_path.exists():
                        lang_stats = pd.read_csv(lang_stats_path)
                        metric_row = lang_stats[lang_stats['metric'] == metric]
                        if len(metric_row) > 0:
                            row[f'{metric}_std'] = metric_row.iloc[0]['std']
                            row[f'{metric}_min'] = metric_row.iloc[0]['min']
                            row[f'{metric}_max'] = metric_row.iloc[0]['max']
                            row['sample_count'] = metric_row.iloc[0]['count']
            
            rows.append(row)
        
        all_scores_df = pd.DataFrame(rows)
        all_scores_path = self.results_dir / 'all_scores.csv'
        all_scores_df.to_csv(all_scores_path, index=False)
        logger.info(f"💾 Saved overall scores: {all_scores_path}")


@click.command()
@click.option('--pipeline', '-p', multiple=True, required=True, help='Pipeline ID(s) - space separated (e.g., pipeline_1 pipeline_2)')
@click.option('--languages', '-l', multiple=True, help='Languages to evaluate - space separated (default: all supported by pipeline)')
@click.option('--execution-id', '-e', help='Execution ID (auto-generated if not provided)')
@click.option('--limit', type=int, help='Limit number of samples (for testing)')
@click.option('--output-dir', type=click.Path(), help='Output directory (default: ./results)')
def main(pipeline: tuple, languages: tuple, execution_id: str, limit: int, output_dir: str):
    """
    Evaluate one or more pipelines and generate CSV files with metrics.

    Examples:

        \b
        # Evaluate single pipeline for all languages
        uv run evaluate_pipelines.py --pipeline pipeline_1

        \b
        # Evaluate multiple pipelines for multiple languages
        uv run evaluate_pipelines.py --pipeline pipeline_1 pipeline_2 pipeline_9 --languages efik igbo --limit 2

        \b
        # Evaluate with custom execution ID (for resuming)
        uv run evaluate_pipelines.py --pipeline pipeline_1 --execution-id 20251214_143022

        \b
        # Evaluate specific languages only
        uv run evaluate_pipelines.py --pipeline pipeline_9 --languages efik

        \b
        # Test with limited samples
        uv run evaluate_pipelines.py --pipeline pipeline_1 --limit 10
    """
    try:
        output_path = Path(output_dir) if output_dir else None
        
        # Evaluate each pipeline
        for pipeline_id in pipeline:
            logger.info(f"\n{'='*70}")
            logger.info(f"EVALUATING PIPELINE: {pipeline_id}")
            logger.info(f"{'='*70}\n")
            
            evaluator = PipelineEvaluator(pipeline_id, execution_id, output_path)
            lang_list = list(languages) if languages else None
            evaluator.evaluate_all(lang_list, limit)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
