#!/usr/bin/env python3
"""
Main Pipeline Comparison Script

Orchestrates the complete pipeline evaluation and comparison workflow:
1. Evaluate all 8 pipelines across 4 languages (32 evaluations)
2. Aggregate and compare results
3. Generate visualizations
4. Create interactive dashboard

Usage:
    python run_pipeline_comparison.py [--languages LANG1 LANG2...] [--pipelines PIPE1 PIPE2...]
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add module directories to path
sys.path.insert(0, str(Path(__file__).parent / 'config'))
sys.path.insert(0, str(Path(__file__).parent / 'orchestrator'))
sys.path.insert(0, str(Path(__file__).parent / 'comparator'))
sys.path.insert(0, str(Path(__file__).parent / 'visualizations'))

from pipeline_config import LANGUAGES, get_all_pipeline_ids
from pipeline_orchestrator import PipelineOrchestrator
from pipeline_comparator import PipelineComparator
from pipeline_viz import create_all_visualizations
from dashboard_generator import DashboardGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run complete pipeline comparison evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all pipelines and languages
  python run_pipeline_comparison.py

  # Evaluate specific languages
  python run_pipeline_comparison.py --languages efik igbo

  # Evaluate specific pipelines
  python run_pipeline_comparison.py --pipelines pipeline_1 pipeline_2

  # Custom output directory and execution ID
  python run_pipeline_comparison.py --output-dir ./my_results --execution-id test_run_001
        """
    )

    parser.add_argument(
        '--languages',
        nargs='+',
        choices=list(LANGUAGES.keys()),
        help='Languages to evaluate (default: all)'
    )

    parser.add_argument(
        '--pipelines',
        nargs='+',
        choices=get_all_pipeline_ids(),
        help='Pipeline IDs to evaluate (default: all)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent / 'results',
        help='Output directory for results (default: ./results)'
    )

    parser.add_argument(
        '--execution-id',
        type=str,
        help='Custom execution ID (default: auto-generated timestamp)'
    )

    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation step (use existing results)'
    )

    parser.add_argument(
        '--skip-visualizations',
        action='store_true',
        help='Skip visualization generation'
    )

    parser.add_argument(
        '--skip-dashboard',
        action='store_true',
        help='Skip dashboard generation'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of samples per evaluation (for testing)'
    )

    args = parser.parse_args()

    # Print banner
    print("\n" + "="*70)
    print("PIPELINE COMPARISON EVALUATION SYSTEM")
    print("="*70 + "\n")

    start_time = datetime.now()

    try:
        # Step 1: Evaluation
        if not args.skip_evaluation:
            logger.info("="*70)
            logger.info("STEP 1: EVALUATING PIPELINES")
            logger.info("="*70)

            orchestrator = PipelineOrchestrator(
                output_dir=args.output_dir,
                execution_id=args.execution_id
            )

            results = orchestrator.evaluate_all_pipelines(
                languages=args.languages,
                pipelines=args.pipelines,
                limit=args.limit
            )

            # Generate manifest
            manifest = orchestrator.generate_manifest()

            logger.info(f"\n✓ Evaluation complete: {len(results)} successful evaluations")
            results_dir = orchestrator.results_dir
        else:
            logger.info("Skipping evaluation step (using existing results)")
            # Find most recent results directory if no execution-id specified
            if args.execution_id:
                results_dir = args.output_dir / args.execution_id
            else:
                # Find most recent
                results_dirs = sorted([d for d in args.output_dir.iterdir() if d.is_dir()],
                                     key=lambda x: x.stat().st_mtime, reverse=True)
                if not results_dirs:
                    logger.error("No existing results found!")
                    return 1
                results_dir = results_dirs[0]
                logger.info(f"Using results from: {results_dir}")

        # Step 2: Comparison
        logger.info("\n" + "="*70)
        logger.info("STEP 2: COMPARING PIPELINES")
        logger.info("="*70)

        comparator = PipelineComparator(results_dir)
        comparator.load_all_results()

        # Create comparison table
        comparison_df = comparator.create_comparison_table()
        logger.info(f"Created comparison table: {len(comparison_df)} rows")

        # Generate rankings per metric
        bleu_rankings = comparator.rank_pipelines(by_language=False, metric='bleu')
        mcd_rankings = comparator.rank_pipelines(by_language=False, metric='mcd')
        language_rankings = comparator.rank_pipelines(by_language=True, metric='bleu')

        # Combine rankings for visualization
        overall_rankings = bleu_rankings  # Use BLEU as primary for visualizations

        # Identify best performers
        best_info = comparator.identify_best_pipeline()

        # Save comparison results
        comparisons_dir = results_dir / 'comparisons'
        comparator.save_comparison_results(comparisons_dir)

        logger.info(f"\n✓ Comparison complete")

        # Step 3: Visualizations
        if not args.skip_visualizations:
            logger.info("\n" + "="*70)
            logger.info("STEP 3: GENERATING VISUALIZATIONS")
            logger.info("="*70)

            viz_dir = results_dir / 'visualizations'
            create_all_visualizations(
                comparison_df=comparison_df,
                rankings=overall_rankings,
                best_info=best_info,
                output_dir=viz_dir
            )

            logger.info(f"\n✓ Visualizations complete")
        else:
            logger.info("\nSkipping visualization generation")

        # Step 4: Dashboard
        if not args.skip_dashboard:
            logger.info("\n" + "="*70)
            logger.info("STEP 4: GENERATING DASHBOARD")
            logger.info("="*70)

            generator = DashboardGenerator(results_dir)
            dashboard_path = generator.generate_dashboard()

            logger.info(f"\n✓ Dashboard complete")
        else:
            logger.info("\nSkipping dashboard generation")
            dashboard_path = None

        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "="*70)
        print("PIPELINE COMPARISON COMPLETE!")
        print("="*70)
        print(f"\nDuration: {duration}")
        print(f"\nResults directory: {results_dir}")
        print(f"  - Individual results: {results_dir / 'individual_pipelines'}")
        print(f"  - Comparisons: {results_dir / 'comparisons'}")
        print(f"  - Visualizations: {results_dir / 'visualizations'}")
        if dashboard_path:
            print(f"  - Dashboard: {dashboard_path}")

        print(f"\n🏆 Best Pipelines by Metric:")
        for metric, info in best_info['by_metric'].items():
            print(f"  {metric.upper()}: {info['pipeline_name']} ({info['score']:.2f}) in {info['language']}")

        print(f"\nBest by Language (using BLEU):")
        for lang, info in best_info['by_language'].items():
            print(f"  {lang.capitalize()}: {info['pipeline_name']} ({info['primary_metric'].upper()}={info['score']:.2f})")

        # Show metric rankings
        print(f"\nTop 3 Pipelines by BLEU (avg across languages):")
        for i, item in enumerate(bleu_rankings['bleu'][:3], 1):
            print(f"  {i}. {item['pipeline_name']}: {item['avg_score']:.2f}")

        print(f"\nTop 3 Pipelines by MCD (avg across languages, lower is better):")
        for i, item in enumerate(mcd_rankings['mcd'][:3], 1):
            print(f"  {i}. {item['pipeline_name']}: {item['avg_score']:.2f} dB")

        if dashboard_path:
            print(f"\n📊 View interactive dashboard: file://{dashboard_path.absolute()}")

        print("\n" + "="*70 + "\n")

        return 0

    except KeyboardInterrupt:
        logger.error("\n\nInterrupted by user")
        return 130

    except Exception as e:
        logger.error(f"\n\nError: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
