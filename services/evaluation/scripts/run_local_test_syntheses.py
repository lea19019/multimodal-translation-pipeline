#!/usr/bin/env python3
"""
Local Test Synthesis Script

Run all 8 synthesis pipelines with a small number of samples (1-2 per language)
on CPU for testing purposes. This is useful when GPU access is limited.

Usage:
    python run_local_test_syntheses.py --n-samples 2
"""

import argparse
import pandas as pd
from pathlib import Path
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import pipeline configs from synthesis orchestrator
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_all_syntheses import PIPELINE_CONFIGS, LANGUAGES, TTS_MODELS, format_text_for_pipeline

def run_local_synthesis(
    pipeline_config: dict,
    iso_code: str,
    selected_samples_path: Path,
    n_samples: int,
    data_base_dir: Path,
    tts_venv_path: Path
):
    """Run synthesis locally on CPU for testing."""

    pipeline_id = pipeline_config['id']
    language_name = LANGUAGES[iso_code]
    model_path = TTS_MODELS[pipeline_config['tts_model']]
    descriptor = pipeline_config['descriptor']

    logger.info(f"\nPipeline {pipeline_id} - {language_name}: {pipeline_config['name']}")

    # Load and format samples
    df = pd.read_csv(selected_samples_path, sep='|')
    df = df.head(n_samples)  # Limit to n samples
    df = format_text_for_pipeline(df, pipeline_config, iso_code)

    # Create temporary CSV
    temp_csv = Path(f"/tmp/test_pipeline_{pipeline_id}_{iso_code}.csv")
    df.to_csv(temp_csv, sep='|', index=False)

    logger.info(f"  Synthesizing {len(df)} samples...")
    logger.info(f"  TTS Model: {pipeline_config['tts_model']}")
    logger.info(f"  Text format: {pipeline_config['text_format']}")

    # Run batch_synthesize.py
    cmd = [
        'uv', 'run', 'python', 'batch_synthesize.py',
        '--csv-file', str(temp_csv),
        '--output-dir', str(data_base_dir),
        '--language', iso_code,
        '--model-path', model_path,
        '--descriptor', descriptor,
        '--device', 'cpu',
        '--sample-rate', '16000',
        '--max-samples', str(n_samples),
        '--text-column', 'text'
    ]

    try:
        # Change to TTS directory and run
        result = subprocess.run(
            cmd,
            cwd='/home/vacl2/multimodal_translation/services/tts',
            capture_output=False,  # Show output in real-time for debugging
            text=True,
            timeout=600  # 10 minutes timeout per synthesis
        )

        if result.returncode == 0:
            logger.info(f"  ✓ Synthesis completed successfully")
        else:
            logger.error(f"  ✗ Synthesis failed with return code {result.returncode}")

    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ Synthesis timed out after 10 minutes")
    except Exception as e:
        logger.error(f"  ✗ Synthesis failed: {e}")
    finally:
        # Clean up temp file
        if temp_csv.exists():
            temp_csv.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Run local CPU-based synthesis test for all 8 pipelines"
    )

    parser.add_argument(
        '--n-samples',
        type=int,
        default=2,
        help='Number of samples to synthesize per language (default: 2)'
    )
    parser.add_argument(
        '--eval-dir',
        type=str,
        default='/home/vacl2/multimodal_translation/services/evaluation',
        help='Evaluation directory containing selected samples'
    )
    parser.add_argument(
        '--data-base-dir',
        type=str,
        default='/home/vacl2/multimodal_translation/services/data/languages',
        help='Base directory for language data (synthesis output destination)'
    )
    parser.add_argument(
        '--tts-venv',
        type=str,
        default='/home/vacl2/multimodal_translation/services/tts/.venv',
        help='Path to TTS virtual environment'
    )
    parser.add_argument(
        '--pipelines',
        nargs='+',
        type=int,
        choices=range(1, 9),
        default=list(range(1, 9)),
        help='Which pipelines to run (default: all 8)'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        choices=list(LANGUAGES.keys()),
        default=list(LANGUAGES.keys()),
        help='Which languages to process (default: all 4)'
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    data_base_dir = Path(args.data_base_dir)
    tts_venv_path = Path(args.tts_venv)

    logger.info("="*70)
    logger.info("LOCAL TEST SYNTHESIS (CPU)")
    logger.info("="*70)
    logger.info(f"Samples per language: {args.n_samples}")
    logger.info(f"Pipelines: {args.pipelines}")
    logger.info(f"Languages: {[LANGUAGES[iso] for iso in args.languages]}")
    logger.info(f"Total syntheses: {len(args.pipelines) * len(args.languages)}")
    logger.info("")
    logger.info("WARNING: Running on CPU - this will be SLOW!")
    logger.info("Each sample may take 30-60 seconds to synthesize.")
    logger.info("")

    # Filter pipelines
    pipelines_to_run = [p for p in PIPELINE_CONFIGS if p['id'] in args.pipelines]

    # Run synthesis for each pipeline/language combination
    total_count = 0
    success_count = 0

    for pipeline_config in pipelines_to_run:
        for iso_code in args.languages:
            language_name = LANGUAGES[iso_code]

            # Check if selected samples exist
            selected_samples_path = eval_dir / f"selected_samples_{iso_code}.csv"
            if not selected_samples_path.exists():
                logger.error(f"Selected samples not found: {selected_samples_path}")
                logger.error("Run select_synthesis_samples.py first!")
                continue

            total_count += 1

            try:
                run_local_synthesis(
                    pipeline_config=pipeline_config,
                    iso_code=iso_code,
                    selected_samples_path=selected_samples_path,
                    n_samples=args.n_samples,
                    data_base_dir=data_base_dir,
                    tts_venv_path=tts_venv_path
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Failed: Pipeline {pipeline_config['id']} - {language_name}: {e}")

    logger.info("\n" + "="*70)
    logger.info("LOCAL TEST SYNTHESIS COMPLETE")
    logger.info("="*70)
    logger.info(f"Completed: {success_count}/{total_count} syntheses")
    logger.info(f"\nOutput directories:")
    logger.info(f"  {data_base_dir}/{{language}}/predicted_{{descriptor}}_{{model_checkpoint}}/")
    logger.info(f"\nMetadata CSVs:")
    logger.info(f"  {data_base_dir}/{{language}}/predicted_{{descriptor}}_{{model_checkpoint}}.csv")


if __name__ == '__main__':
    main()
