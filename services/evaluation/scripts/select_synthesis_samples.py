#!/usr/bin/env python3
"""
Sample Selection Script for Multi-Pipeline TTS Synthesis

This script randomly selects N samples per language from the NLLB predictions CSV
for use across all 8 synthesis pipelines. The same samples are used for all pipelines
to ensure fair comparison.

Usage:
    python select_synthesis_samples.py --n-samples 300 --seed 42 --output-dir ./

Output:
    Creates selected_samples_{iso_code}.csv for each language
"""

import argparse
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Language mappings
LANGUAGES = {
    'efi': 'efik',
    'ibo': 'igbo',
    'swa': 'swahili',
    'xho': 'xhosa'
}

def select_samples_for_language(
    language_name: str,
    iso_code: str,
    data_base_dir: Path,
    n_samples: int,
    seed: int
) -> pd.DataFrame:
    """
    Select N random samples for a given language.

    Args:
        language_name: Full language name (e.g., 'efik')
        iso_code: ISO language code (e.g., 'efi')
        data_base_dir: Base directory containing language data
        n_samples: Number of samples to select
        seed: Random seed for reproducibility

    Returns:
        DataFrame with selected samples
    """
    # Path to NMT predictions CSV
    nmt_csv_path = data_base_dir / language_name / "nmt_predictions_multilang_finetuned_final.csv"

    logger.info(f"\nProcessing {language_name} ({iso_code})")
    logger.info(f"  Reading: {nmt_csv_path}")

    if not nmt_csv_path.exists():
        raise FileNotFoundError(f"NMT predictions not found: {nmt_csv_path}")

    # Load NMT predictions
    df = pd.read_csv(nmt_csv_path, sep='|')
    total_available = len(df)

    logger.info(f"  Total available samples: {total_available}")

    if total_available < n_samples:
        logger.warning(f"  Only {total_available} samples available, requested {n_samples}")
        logger.warning(f"  Selecting all {total_available} samples")
        selected = df
    else:
        # Randomly select N samples with fixed seed
        selected = df.sample(n=n_samples, random_state=seed)
        logger.info(f"  Selected {len(selected)} samples (seed={seed})")

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Select random samples for multi-pipeline TTS synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--n-samples',
        type=int,
        default=300,
        help='Number of samples to select per language (default: 300)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--data-base-dir',
        type=str,
        default='/home/vacl2/multimodal_translation/services/data/languages',
        help='Base directory containing language data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/vacl2/multimodal_translation/services/evaluation',
        help='Output directory for selected samples CSVs'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        choices=list(LANGUAGES.keys()),
        default=list(LANGUAGES.keys()),
        help='Language ISO codes to process (default: all)'
    )

    args = parser.parse_args()

    data_base_dir = Path(args.data_base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("SAMPLE SELECTION FOR MULTI-PIPELINE TTS SYNTHESIS")
    logger.info("="*70)
    logger.info(f"Samples per language: {args.n_samples}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Data directory: {data_base_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Languages: {', '.join([LANGUAGES[iso] for iso in args.languages])}")

    # Process each language
    total_selected = 0

    for iso_code in args.languages:
        language_name = LANGUAGES[iso_code]

        try:
            # Select samples
            selected = select_samples_for_language(
                language_name=language_name,
                iso_code=iso_code,
                data_base_dir=data_base_dir,
                n_samples=args.n_samples,
                seed=args.seed
            )

            # Save to CSV
            output_path = output_dir / f"selected_samples_{iso_code}.csv"
            selected.to_csv(output_path, sep='|', index=False)
            logger.info(f"  Saved: {output_path}")
            logger.info(f"  Columns: {', '.join(selected.columns.tolist())}")

            total_selected += len(selected)

        except Exception as e:
            logger.error(f"  Failed to process {language_name}: {e}")
            continue

    logger.info("\n" + "="*70)
    logger.info("SELECTION COMPLETE")
    logger.info("="*70)
    logger.info(f"Total samples selected: {total_selected}")
    logger.info(f"Output files:")
    for iso_code in args.languages:
        output_file = output_dir / f"selected_samples_{iso_code}.csv"
        if output_file.exists():
            logger.info(f"  - {output_file}")

    logger.info("\nThese samples will be used across all 8 synthesis pipelines.")
    logger.info("Next step: Run synthesis orchestrator to generate audio for each pipeline.")


if __name__ == '__main__':
    main()
