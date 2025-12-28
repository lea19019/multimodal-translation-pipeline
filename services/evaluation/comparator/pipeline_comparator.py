#!/usr/bin/env python3
"""
Pipeline Comparator

Aggregates and compares evaluation results across all pipelines.
Generates rankings, identifies best performers, and computes statistics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class PipelineComparator:
    """
    Aggregates and compares pipeline evaluation results.

    Provides methods to:
    - Create comparison tables
    - Rank pipelines by various criteria
    - Identify best-performing pipelines
    - Compute statistical comparisons
    """

    def __init__(self, results_dir: Path):
        """
        Initialize comparator.

        Args:
            results_dir: Directory containing individual pipeline results
        """
        self.results_dir = Path(results_dir)
        self.individual_dir = self.results_dir / 'individual_pipelines'
        self.all_results = {}

        logger.info(f"Initialized comparator for: {results_dir}")

    def load_all_results(self) -> Dict:
        """
        Load all individual pipeline results from disk.

        Returns:
            Dict mapping (pipeline_id, language) -> results
        """
        logger.info("Loading all individual results...")

        all_results = {}

        if not self.individual_dir.exists():
            logger.error(f"Individual results directory not found: {self.individual_dir}")
            return all_results

        # Iterate through pipeline directories
        for pipeline_dir in self.individual_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue

            pipeline_short_name = pipeline_dir.name

            # Iterate through language directories
            for lang_dir in pipeline_dir.iterdir():
                if not lang_dir.is_dir():
                    continue

                language = lang_dir.name
                metrics_file = lang_dir / 'metrics.json'

                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        result = json.load(f)
                        pipeline_id = result['pipeline_id']
                        key = f"{pipeline_id}_{language}"
                        all_results[key] = result

        logger.info(f"Loaded {len(all_results)} individual results")
        self.all_results = all_results
        return all_results

    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create comprehensive comparison table.

        Returns:
            DataFrame with columns:
            - pipeline_id, pipeline_name, language
            - bleu, chrf, comet (corpus scores)
            - mcd, blaser (corpus scores)
        """
        logger.info("Creating comparison table...")

        if not self.all_results:
            self.load_all_results()

        rows = []

        for key, result in self.all_results.items():
            row = {
                'pipeline_id': result['pipeline_id'],
                'pipeline_name': result['pipeline_name'],
                'language': result['language'],
                'n_samples': result['n_samples'],
                'uses_nmt': result['uses_nmt']
            }

            # Extract metric scores
            metrics = result.get('metrics', {})

            # Text metrics
            row['bleu'] = metrics.get('bleu', {}).get('corpus_score') if metrics.get('bleu') else None
            row['chrf'] = metrics.get('chrf', {}).get('corpus_score') if metrics.get('chrf') else None
            row['comet'] = metrics.get('comet', {}).get('corpus_score') if metrics.get('comet') else None

            # Audio metrics
            row['mcd'] = metrics.get('mcd', {}).get('mean_mcd') if metrics.get('mcd') else None
            row['blaser'] = metrics.get('blaser', {}).get('corpus_score') if metrics.get('blaser') else None

            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Created comparison table with {len(df)} rows")

        return df

    def rank_pipelines(self, by_language: bool = False, metric: str = 'bleu') -> Dict:
        """
        Rank pipelines by performance for a specific metric.

        Args:
            by_language: If True, rank separately for each language
            metric: Metric to rank by ('bleu', 'chrf', 'comet', 'mcd', 'blaser')

        Returns:
            Dict with rankings
        """
        logger.info(f"Ranking pipelines by {metric}...")

        df = self.create_comparison_table()

        # Determine sort order (MCD: lower is better, all others: higher is better)
        ascending = (metric == 'mcd')

        if by_language:
            # Rank within each language
            rankings = {}
            for language in df['language'].unique():
                lang_df = df[df['language'] == language].copy()
                lang_df = lang_df.sort_values(metric, ascending=ascending, na_position='last')

                rankings[language] = [
                    {
                        'rank': i + 1,
                        'pipeline_id': row['pipeline_id'],
                        'pipeline_name': row['pipeline_name'],
                        'score': row[metric]
                    }
                    for i, (_, row) in enumerate(lang_df.iterrows())
                ]

            logger.info(f"Ranked pipelines for {len(rankings)} languages by {metric}")
            return rankings
        else:
            # Overall ranking (average across languages)
            avg_scores = df.groupby(['pipeline_id', 'pipeline_name'])[metric].mean()
            avg_scores = avg_scores.sort_values(ascending=ascending)

            rankings = [
                {
                    'rank': i + 1,
                    'pipeline_id': pipeline_id,
                    'pipeline_name': pipeline_name,
                    'avg_score': score
                }
                for i, ((pipeline_id, pipeline_name), score) in enumerate(avg_scores.items())
            ]

            logger.info(f"{metric.upper()} ranking: {rankings[0]['pipeline_name']} is #1 (avg: {rankings[0]['avg_score']:.2f})")
            return {metric: rankings}

    def identify_best_pipeline(self) -> Dict:
        """
        Identify best-performing pipelines by metric.

        Returns:
            Dict with:
            - by_metric: Best pipeline for each metric (across all languages)
            - by_language: Best pipeline for each language (per metric)
        """
        logger.info("Identifying best pipelines...")

        df = self.create_comparison_table()

        result = {
            'by_metric': {},
            'by_language': {}
        }

        # Best by metric (across all languages)
        metrics = ['bleu', 'chrf', 'comet', 'mcd', 'blaser']
        for metric in metrics:
            if metric in df.columns and df[metric].notna().any():
                # For MCD, lower is better
                if metric == 'mcd':
                    best_idx = df[metric].idxmin()
                    best_value = df[metric].min()
                else:
                    best_idx = df[metric].idxmax()
                    best_value = df[metric].max()

                best_row = df.loc[best_idx]
                result['by_metric'][metric] = {
                    'pipeline_id': best_row['pipeline_id'],
                    'pipeline_name': best_row['pipeline_name'],
                    'language': best_row['language'],
                    'score': best_value
                }

        # Best by language (using BLEU as primary metric for NMT pipelines, MCD for others)
        for language in df['language'].unique():
            lang_df = df[df['language'] == language]

            # Prefer BLEU if available, otherwise MCD
            if lang_df['bleu'].notna().any():
                best_idx = lang_df['bleu'].idxmax()
                best_row = lang_df.loc[best_idx]
                primary_metric = 'bleu'
                score = best_row['bleu']
            elif lang_df['mcd'].notna().any():
                best_idx = lang_df['mcd'].idxmin()  # Lower is better
                best_row = lang_df.loc[best_idx]
                primary_metric = 'mcd'
                score = best_row['mcd']
            else:
                continue

            result['by_language'][language] = {
                'pipeline_id': best_row['pipeline_id'],
                'pipeline_name': best_row['pipeline_name'],
                'primary_metric': primary_metric,
                'score': score
            }

        logger.info(f"Identified best pipelines for {len(result['by_metric'])} metrics and {len(result['by_language'])} languages")
        return result

    def save_comparison_results(self, output_dir: Path):
        """
        Save comparison results to files.

        Args:
            output_dir: Directory to save comparison results
        """
        logger.info(f"Saving comparison results to: {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Comparison table (CSV and JSON)
        df = self.create_comparison_table()

        csv_path = output_dir / 'cross_pipeline_stats.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"  ✓ Saved: {csv_path}")

        json_path = output_dir / 'all_pipelines_summary.json'
        with open(json_path, 'w') as f:
            json.dump(df.to_dict(orient='records'), f, indent=2, default=str)
        logger.info(f"  ✓ Saved: {json_path}")

        # 2. Rankings by metric
        bleu_rankings = self.rank_pipelines(by_language=False, metric='bleu')
        mcd_rankings = self.rank_pipelines(by_language=False, metric='mcd')

        rankings_path = output_dir / 'pipeline_rankings_overall.json'
        with open(rankings_path, 'w') as f:
            json.dump({'bleu': bleu_rankings, 'mcd': mcd_rankings}, f, indent=2)
        logger.info(f"  ✓ Saved: {rankings_path}")

        # 3. Per-language rankings
        language_rankings = self.rank_pipelines(by_language=True, metric='bleu')

        lang_rankings_path = output_dir / 'pipeline_rankings_by_language.json'
        with open(lang_rankings_path, 'w') as f:
            json.dump(language_rankings, f, indent=2)
        logger.info(f"  ✓ Saved: {lang_rankings_path}")

        # 4. Best performers
        best_info = self.identify_best_pipeline()

        best_path = output_dir / 'best_pipelines.json'
        with open(best_path, 'w') as f:
            json.dump(best_info, f, indent=2)
        logger.info(f"  ✓ Saved: {best_path}")

        # 5. Per-metric rankings
        for metric in ['bleu', 'chrf', 'comet', 'mcd', 'blaser']:
            if metric in df.columns and df[metric].notna().any():
                metric_rankings = self.rank_pipelines(by_language=False, metric=metric)

                metric_path = output_dir / f'rankings_by_{metric}.json'
                with open(metric_path, 'w') as f:
                    json.dump(metric_rankings, f, indent=2)
                logger.info(f"  ✓ Saved: {metric_path}")

        logger.info("✓ All comparison results saved")


if __name__ == '__main__':
    # Test the comparator
    import argparse

    parser = argparse.ArgumentParser(description="Test pipeline comparator")
    parser.add_argument('--results-dir', type=Path, required=True,
                        help='Path to results directory (e.g., results/pipeline_comparison_20251212_143022)')
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        exit(1)

    # Initialize comparator
    comparator = PipelineComparator(args.results_dir)

    # Load all results
    all_results = comparator.load_all_results()
    print(f"\nLoaded {len(all_results)} individual results")

    # Create comparison table
    df = comparator.create_comparison_table()
    print(f"\nComparison Table Preview:")
    print(df.head(10).to_string())

    # Rankings
    print("\n" + "="*70)
    print("OVERALL RANKINGS")
    print("="*70)
    overall_rankings = comparator.rank_pipelines(by_language=False)
    for item in overall_rankings['overall'][:5]:
        print(f"{item['rank']}. {item['pipeline_name']}: {item['avg_score']:.4f}")

    # Best performers
    print("\n" + "="*70)
    print("BEST PERFORMERS")
    print("="*70)
    best_info = comparator.identify_best_pipeline()
    print(f"\nOverall Best: {best_info['overall_best']['pipeline_name']}")
    print(f"  Score: {best_info['overall_best']['score']:.4f}")

    print("\nBest by Language:")
    for lang, info in best_info['by_language'].items():
        print(f"  {lang.capitalize()}: {info['pipeline_name']} ({info['score']:.4f})")

    # Save results
    comparisons_dir = args.results_dir / 'comparisons'
    comparator.save_comparison_results(comparisons_dir)

    print(f"\n✓ Comparison results saved to: {comparisons_dir}")
