#!/usr/bin/env python3
"""
Pipeline Evaluation Orchestrator

Coordinates evaluation of all 8 pipelines across 4 languages (32 total evaluations).
Reuses metric computation logic from existing evaluation scripts.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys
import pandas as pd

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent.parent / 'scripts'
sys.path.insert(0, str(scripts_dir))

# Add config directory to path
config_dir = Path(__file__).parent.parent / 'config'
sys.path.insert(0, str(config_dir))

from pipeline_config import (
    get_pipeline, get_all_pipeline_ids, get_synthesis_path,
    get_available_syntheses, LANGUAGES
)
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


class PipelineOrchestrator:
    """
    Orchestrates evaluation of all pipelines across languages.

    Evaluates each pipeline × language combination and saves results in
    organized directory structure.
    """

    def __init__(self, output_dir: Path = None, execution_id: str = None):
        """
        Initialize orchestrator.

        Args:
            output_dir: Base output directory (default: services/evaluation/results)
            execution_id: Execution identifier (auto-generated if None)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'results'

        self.output_dir = Path(output_dir)

        if execution_id is None:
            execution_id = f"pipeline_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.execution_id = execution_id
        self.results_dir = self.output_dir / execution_id

        # Create directory structure
        self.individual_dir = self.results_dir / 'individual_pipelines'
        self.comparisons_dir = self.results_dir / 'comparisons'
        self.visualizations_dir = self.results_dir / 'visualizations'

        # Create all directories
        for d in [self.individual_dir, self.comparisons_dir, self.visualizations_dir]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized orchestrator: {self.execution_id}")
        logger.info(f"Results directory: {self.results_dir}")

    def evaluate_all_pipelines(self, languages: List[str] = None, pipelines: List[str] = None, limit: int = None) -> Dict:
        """
        Evaluate all pipelines across specified languages.

        Args:
            languages: List of language names (default: all 4 languages)
            pipelines: List of pipeline IDs (default: all 8 pipelines)
            limit: Limit number of samples per evaluation (for testing)

        Returns:
            Dict mapping (pipeline_id, language) -> evaluation results
        """
        self.sample_limit = limit
        if languages is None:
            languages = list(LANGUAGES.keys())

        if pipelines is None:
            pipelines = get_all_pipeline_ids()

        total = len(pipelines) * len(languages)
        logger.info(f"Starting evaluation of {len(pipelines)} pipelines × {len(languages)} languages = {total} evaluations")

        all_results = {}
        count = 0

        for pipeline_id in pipelines:
            pipeline = get_pipeline(pipeline_id)

            for language in languages:
                count += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"[{count}/{total}] Evaluating: {pipeline['name']} × {language}")
                logger.info(f"{'='*70}")

                try:
                    # Evaluate this combination
                    result = self._evaluate_single(pipeline_id, language)

                    if result:
                        # Store in results dict
                        key = f"{pipeline_id}_{language}"
                        all_results[key] = result

                        # Save individual result
                        self._save_individual_result(pipeline, language, result)

                        logger.info(f"✓ Completed: {pipeline['name']} × {language}")
                    else:
                        logger.warning(f"⚠ No results for: {pipeline['name']} × {language}")

                except Exception as e:
                    logger.error(f"✗ Failed: {pipeline['name']} × {language}")
                    logger.error(f"  Error: {e}", exc_info=True)

        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Successfully evaluated: {len(all_results)}/{total} combinations")

        return all_results

    def _evaluate_single(self, pipeline_id: str, language: str) -> Optional[Dict]:
        """
        Evaluate a single pipeline × language combination.

        Args:
            pipeline_id: Pipeline identifier
            language: Language name

        Returns:
            Evaluation results dict or None if evaluation failed
        """
        # Get paths for this pipeline/language
        paths = get_synthesis_path(pipeline_id, language)
        pipeline = paths['pipeline']

        csv_path = paths['csv_path']
        nmt_csv_path = paths['nmt_csv_path']
        audio_dir = paths['audio_dir']
        ref_audio_dir = paths['ref_audio_dir']
        iso_code = paths['iso_code']

        # Verify paths exist
        if not csv_path.exists():
            logger.error(f"CSV not found: {csv_path}")
            return None

        # Load data
        logger.info(f"Loading predicted samples: {csv_path.name}")
        predicted_df = pd.read_csv(csv_path, sep='|')
        logger.info(f"  Rows: {len(predicted_df)}")

        logger.info(f"Loading ground truth NMT data: {nmt_csv_path.name}")
        nmt_df = pd.read_csv(nmt_csv_path, sep='|')
        logger.info(f"  Rows: {len(nmt_df)}")

        # Merge on segment_id
        logger.info("Merging data on segment_id...")
        merged_df = predicted_df.merge(
            nmt_df[['segment_id', 'ground_truth_tgt_text', 'src_text', 'predicted_tgt_text']],
            on='segment_id',
            how='left'
        )
        logger.info(f"Merged rows: {len(merged_df)}")

        # Check for duplicates
        duplicates = merged_df['segment_id'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"⚠️  Found {duplicates} duplicate segment_ids in merged data")

        # Filter successful syntheses
        successful = merged_df[merged_df['success'] == True].copy()
        logger.info(f"Successful syntheses before dedup: {len(successful)} / {len(merged_df)}")

        # Remove duplicates (keep first occurrence)
        successful = successful.drop_duplicates(subset=['segment_id'], keep='first')
        logger.info(f"After removing duplicates: {len(successful)}")

        # Apply sample limit if specified (for testing)
        if hasattr(self, 'sample_limit') and self.sample_limit is not None:
            successful = successful.head(self.sample_limit)
            logger.info(f"Limited to {len(successful)} samples for testing")

        if len(successful) == 0:
            logger.error("No successful syntheses to evaluate!")
            return None

        # Prepare evaluation data
        predictions = successful['text'].tolist()
        references = successful['ground_truth_tgt_text'].tolist()
        sources = successful['src_text'].tolist()

        logger.info(f"Evaluating {len(predictions)} samples...")

        # Initialize result dict
        result = {
            'pipeline_id': pipeline_id,
            'pipeline_name': pipeline['name'],
            'language': language,
            'iso_code': iso_code,
            'n_samples': len(successful),
            'uses_nmt': pipeline['uses_nmt'],
            'metrics': {}
        }

        # Compute text metrics (only if pipeline uses NMT)
        if pipeline['uses_nmt']:
            logger.info("\n" + "="*70)
            logger.info("TEXT METRICS")
            logger.info("="*70)

            text_metrics = TextMetrics(chrf_word_order=2)  # chrF++ with word order

            # BLEU
            if 'bleu' in pipeline['metrics']:
                try:
                    logger.info("\n📊 Computing BLEU...")
                    bleu_results = text_metrics.compute_bleu(
                        hypotheses=predictions,
                        references=[[ref] for ref in references]
                    )
                    result['metrics']['bleu'] = bleu_results
                    logger.info(f"  Corpus BLEU: {bleu_results['corpus_score']:.2f}")
                except Exception as e:
                    logger.warning(f"BLEU computation failed: {e}")
                    result['metrics']['bleu'] = None

            # chrF++
            if 'chrf' in pipeline['metrics']:
                try:
                    logger.info("\n📊 Computing chrF++...")
                    chrf_results = text_metrics.compute_chrf(
                        hypotheses=predictions,
                        references=[[ref] for ref in references]
                    )
                    result['metrics']['chrf'] = chrf_results
                    logger.info(f"  Corpus chrF++: {chrf_results['corpus_score']:.2f}")
                except Exception as e:
                    logger.warning(f"chrF computation failed: {e}")
                    result['metrics']['chrf'] = None

            # COMET
            if 'comet' in pipeline['metrics']:
                try:
                    logger.info("\n📊 Computing COMET (SSA-COMET-QE for African languages)...")
                    comet = CometEvaluator(model_name="McGill-NLP/ssa-comet-qe")
                    comet_results = comet.evaluate(
                        sources=sources,
                        hypotheses=predictions,
                        references=references
                    )
                    result['metrics']['comet'] = comet_results
                    logger.info(f"  Corpus COMET: {comet_results['corpus_score']:.4f}")
                except Exception as e:
                    logger.warning(f"COMET computation failed: {e}")
                    result['metrics']['comet'] = None
        else:
            logger.info(f"Skipping text metrics (pipeline doesn't use NMT)")

        # Compute audio metrics
        logger.info("\n" + "="*70)
        logger.info("AUDIO METRICS")
        logger.info("="*70)

        if audio_dir.exists() and ref_audio_dir.exists():
            # Build audio file path lists
            generated_audio_paths = []
            reference_audio_paths = []

            for idx, row in successful.iterrows():
                segment_id = row['segment_id']
                user_id = row['user_id']

                gen_audio = audio_dir / f"Segment={segment_id}_User={user_id}_Language={iso_code}_pred.wav"
                ref_audio = ref_audio_dir / f"Segment={segment_id}_User={user_id}_Language={iso_code}.wav"

                if gen_audio.exists() and ref_audio.exists():
                    generated_audio_paths.append(gen_audio)
                    reference_audio_paths.append(ref_audio)

            logger.info(f"Found {len(generated_audio_paths)} audio pairs")

            if generated_audio_paths:
                # MCD
                if 'mcd' in pipeline['metrics']:
                    try:
                        logger.info("\n🎵 Computing MCD (Mel-Cepstral Distance)...")
                        audio_metrics = AudioMetrics()
                        mcd_results = audio_metrics.compute_mcd_batch(
                            generated_audio_paths,
                            reference_audio_paths
                        )
                        result['metrics']['mcd'] = mcd_results
                        logger.info(f"  Mean MCD: {mcd_results['mean_mcd']:.2f} (lower is better)")
                        logger.info(f"  Std MCD:  {mcd_results['std_mcd']:.2f}")
                    except Exception as e:
                        logger.warning(f"MCD computation failed: {e}")
                        result['metrics']['mcd'] = None

                # BLASER
                if 'blaser' in pipeline['metrics']:
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

                        # For BLASER, we need source audio (use reference for now)
                        source_audio_paths = reference_audio_paths.copy()

                        blaser = BlaserEvaluator(model_name="blaser_2_0_qe")
                        blaser_results = blaser.evaluate(
                            source_audio_paths=source_audio_paths,
                            target_audio_paths=generated_audio_paths,
                            reference_texts=references,
                            source_texts=sources,
                            source_lang=target_lang,
                            target_lang=target_lang
                        )
                        result['metrics']['blaser'] = blaser_results
                        logger.info(f"  Corpus BLASER: {blaser_results['corpus_score']:.4f} (0-5, higher is better)")
                    except Exception as e:
                        logger.warning(f"BLASER computation failed: {e}")
                        result['metrics']['blaser'] = None
            else:
                logger.warning("No audio pairs found")
        else:
            logger.warning(f"Audio directories not found")

        return result

    def _save_individual_result(self, pipeline: Dict, language: str, result: Dict):
        """
        Save individual pipeline × language result.

        Args:
            pipeline: Pipeline configuration
            language: Language name
            result: Evaluation results
        """
        # Create directory: individual_pipelines/{pipeline_short_name}/{language}/
        pipeline_dir = self.individual_dir / pipeline['short_name']
        lang_dir = pipeline_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics.json
        metrics_path = lang_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"💾 Saved: {metrics_path}")

        # Save sample scores CSV (if available)
        sample_rows = []
        for metric_name, metric_data in result['metrics'].items():
            if metric_data and 'sentence_scores' in metric_data:
                scores = metric_data['sentence_scores']
                for i, score in enumerate(scores):
                    if i >= len(sample_rows):
                        sample_rows.append({'sample_idx': i})
                    sample_rows[i][f'{metric_name}_score'] = score

        if sample_rows:
            sample_df = pd.DataFrame(sample_rows)
            sample_csv_path = lang_dir / 'sample_scores.csv'
            sample_df.to_csv(sample_csv_path, index=False)
            logger.info(f"💾 Saved: {sample_csv_path}")

    def generate_manifest(self) -> Dict:
        """
        Generate execution manifest with metadata.

        Returns:
            Manifest dict
        """
        manifest = {
            'execution_id': self.execution_id,
            'timestamp': datetime.now().isoformat(),
            'results_directory': str(self.results_dir),
            'total_pipelines': len(get_all_pipeline_ids()),
            'total_languages': len(LANGUAGES),
            'total_evaluations': len(get_all_pipeline_ids()) * len(LANGUAGES)
        }

        manifest_path = self.results_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"💾 Saved manifest: {manifest_path}")
        return manifest


if __name__ == '__main__':
    # Test the orchestrator
    import argparse

    parser = argparse.ArgumentParser(description="Test pipeline orchestrator")
    parser.add_argument('--language', type=str, help='Single language to test')
    parser.add_argument('--pipeline', type=str, help='Single pipeline to test')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be evaluated')
    args = parser.parse_args()

    if args.dry_run:
        print("="*70)
        print("DRY RUN - Would evaluate:")
        print("="*70)

        languages = [args.language] if args.language else list(LANGUAGES.keys())
        pipelines = [args.pipeline] if args.pipeline else get_all_pipeline_ids()

        for pipeline_id in pipelines:
            pipeline = get_pipeline(pipeline_id)
            for language in languages:
                paths = get_synthesis_path(pipeline_id, language)
                print(f"\n✓ {pipeline['name']} × {language}")
                print(f"  CSV: {paths['csv_path'].name}")
                print(f"  Metrics: {', '.join(pipeline['metrics'])}")

        print(f"\nTotal: {len(pipelines)} × {len(languages)} = {len(pipelines) * len(languages)} evaluations")
    else:
        # Run actual evaluation
        orchestrator = PipelineOrchestrator()

        languages = [args.language] if args.language else None
        pipelines = [args.pipeline] if args.pipeline else None

        results = orchestrator.evaluate_all_pipelines(
            languages=languages,
            pipelines=pipelines
        )

        # Generate manifest
        manifest = orchestrator.generate_manifest()

        print(f"\n{'='*70}")
        print("EVALUATION COMPLETE")
        print(f"{'='*70}")
        print(f"Results directory: {orchestrator.results_dir}")
        print(f"Successful evaluations: {len(results)}")
