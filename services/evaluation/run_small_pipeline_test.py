#!/usr/bin/env python3
"""
Small Pipeline Test Script

Tests evaluation of 3 pipelines with 2-3 samples each to validate
visualization improvements.

Usage:
    python run_small_pipeline_test.py --language efik --limit 3
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
        description='Test pipeline comparison with small sample set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 3 samples from efik
  python run_small_pipeline_test.py --language efik --limit 3

  # Test with 2 samples from swahili
  python run_small_pipeline_test.py --language swa --limit 2

  # Test specific pipelines
  python run_small_pipeline_test.py --language efik --limit 3 --pipelines pipeline_1 pipeline_2 pipeline_5
        """
    )

    parser.add_argument(
        '--language',
        required=True,
        choices=list(LANGUAGES.keys()),
        help='Language to evaluate'
    )

    parser.add_argument(
        '--pipelines',
        nargs='+',
        default=['pipeline_1', 'pipeline_2', 'pipeline_5'],
        choices=get_all_pipeline_ids(),
        help='Pipeline IDs to evaluate (default: pipeline_1, pipeline_2, pipeline_5)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=3,
        help='Number of samples per evaluation (default: 3)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent / 'results',
        help='Output directory for results (default: ./results)'
    )

    args = parser.parse_args()

    # Print banner
    print("\n" + "="*70)
    print("SMALL PIPELINE TEST")
    print("="*70)
    print(f"Language: {args.language}")
    print(f"Pipelines: {', '.join(args.pipelines)}")
    print(f"Samples: {args.limit}")
    print("="*70 + "\n")

    start_time = datetime.now()

    try:
        # Step 1: Evaluation
        logger.info("="*70)
        logger.info("STEP 1: EVALUATING PIPELINES")
        logger.info("="*70)

        execution_id = f"small_test_{args.language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        orchestrator = PipelineOrchestrator(
            output_dir=args.output_dir,
            execution_id=execution_id
        )

        results = orchestrator.evaluate_all_pipelines(
            languages=[args.language],
            pipelines=args.pipelines,
            limit=args.limit
        )

        # Generate manifest
        manifest = orchestrator.generate_manifest()

        logger.info(f"\n✓ Evaluation complete: {len(results)} successful evaluations")
        results_dir = orchestrator.results_dir

        # Step 2: Comparison
        logger.info("\n" + "="*70)
        logger.info("STEP 2: COMPARING PIPELINES")
        logger.info("="*70)

        comparator = PipelineComparator(results_dir)
        comparator.load_all_results()

        # Create comparison table
        comparison_df = comparator.create_comparison_table()
        logger.info(f"Created comparison table: {len(comparison_df)} rows")

        # Generate rankings
        overall_rankings = comparator.rank_pipelines(by_language=False, metric='bleu')

        # Identify best performers
        best_info = comparator.identify_best_pipeline()

        # Save comparison results
        comparisons_dir = results_dir / 'comparisons'
        comparator.save_comparison_results(comparisons_dir)

        logger.info(f"\n✓ Comparison complete")

        # Step 3: Visualizations
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

        # Step 4: Dashboard
        logger.info("\n" + "="*70)
        logger.info("STEP 4: GENERATING DASHBOARD")
        logger.info("="*70)

        generator = DashboardGenerator(results_dir)
        dashboard_path = generator.generate_dashboard()

        logger.info(f"\n✓ Dashboard complete")

        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "="*70)
        print("TEST COMPLETE!")
        print("="*70)
        print(f"\nDuration: {duration}")
        print(f"\nResults directory: {results_dir}")
        print(f"  - Individual results: {results_dir / 'individual_pipelines'}")
        print(f"  - Comparisons: {results_dir / 'comparisons'}")
        print(f"  - Visualizations: {results_dir / 'visualizations'}")
        print(f"  - Dashboard: {dashboard_path}")

        print(f"\n🏆 Best Pipeline:")
        if 'overall' in best_info and len(best_info['overall']) > 0:
            best = best_info['overall'][0]
            print(f"  {best['pipeline_name']}")
            print(f"  Average Score: {best.get('avg_score', 'N/A')}")

        print(f"\n📊 Visualizations Generated:")
        viz_files = list(viz_dir.glob('*.png'))
        for f in sorted(viz_files):
            print(f"  - {f.name}")
        
        per_lang_dir = viz_dir / 'per_language'
        if per_lang_dir.exists():
            per_lang_files = list(per_lang_dir.glob('*.png'))
            if per_lang_files:
                print(f"\n  Per-language visualizations:")
                for f in sorted(per_lang_files):
                    print(f"    - {f.name}")

        print(f"\n📊 View dashboard: file://{dashboard_path.absolute()}")
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
