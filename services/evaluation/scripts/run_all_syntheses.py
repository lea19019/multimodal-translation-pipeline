#!/usr/bin/env python3
"""
Synthesis Orchestrator for Multi-Pipeline TTS Evaluation

This script generates and submits 32 SLURM jobs (8 pipelines × 4 languages) for TTS synthesis.
Each pipeline uses a different combination of TTS model and text format.

Usage:
    python run_all_syntheses.py --submit  # Generate and submit jobs
    python run_all_syntheses.py          # Generate scripts only (dry run)

Prerequisites:
    - Run select_synthesis_samples.py first to select 300 samples per language
    - Ensure selected_samples_{iso_code}.csv files exist

Output:
    - SLURM job scripts in slurm_jobs/
    - Formatted input CSVs in formatted_inputs/
    - Audio synthesis output in /home/vacl2/multimodal_translation/services/data/languages/{lang}/predicted_{descriptor}_{checkpoint}/
"""

import argparse
import pandas as pd
from pathlib import Path
import subprocess
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TTS Model Configurations
TTS_MODELS = {
    'MULTILINGUAL_TRAINING': '/home/vacl2/multimodal_translation/services/tts/checkpoints/MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632',
    'Src_Tgt': '/home/vacl2/multimodal_translation/services/tts/checkpoints/Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8',
    'Translate_Src_Tgt': '/home/vacl2/multimodal_translation/services/tts/checkpoints/Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254',
    'Multilingual_Src': '/home/vacl2/multimodal_translation/services/tts/checkpoints/Multilingual_Src_6_12-December-06-2025_10+30PM-b2b8254',
}

# Pipeline Configurations (8 total)
PIPELINE_CONFIGS = [
    {
        'id': 1,
        'name': 'nllb_MULTILINGUAL',
        'tts_model': 'MULTILINGUAL_TRAINING',
        'text_source': 'predicted_tgt_text',  # Use NLLB prediction
        'text_format': 'plain',  # No special formatting
        'descriptor': 'nllb_tgt',
    },
    {
        'id': 2,
        'name': 'nllb_SrcTgt',
        'tts_model': 'Src_Tgt',
        'text_source': 'predicted_tgt_text',
        'text_format': 'plain',
        'descriptor': 'nllb_tgt',
    },
    {
        'id': 3,
        'name': 'src_SrcTgt',
        'tts_model': 'Src_Tgt',
        'text_source': 'src_text',  # Use English source only
        'text_format': 'plain',
        'descriptor': 'src',
    },
    {
        'id': 4,
        'name': 'custom_lang_SrcTgt',
        'tts_model': 'Src_Tgt',
        'text_source': 'both',  # Use both source and prediction
        'text_format': 'custom_lang',  # <eng>{src} <{lang}>{pred}
        'descriptor': 'custom_lang',
    },
    {
        'id': 5,
        'name': 'nllb_TranslateSrcTgt',
        'tts_model': 'Translate_Src_Tgt',
        'text_source': 'predicted_tgt_text',
        'text_format': 'plain',
        'descriptor': 'nllb_tgt',
    },
    {
        'id': 6,
        'name': 'src_TranslateSrcTgt',
        'tts_model': 'Translate_Src_Tgt',
        'text_source': 'src_text',
        'text_format': 'plain',
        'descriptor': 'src',
    },
    {
        'id': 7,
        'name': 'custom_translate_TranslateSrcTgt',
        'tts_model': 'Translate_Src_Tgt',
        'text_source': 'both',
        'text_format': 'custom_translate',  # <translate> <eng>{src} <{lang}>{pred}
        'descriptor': 'custom_translate',
    },
    {
        'id': 8,
        'name': 'src_MultilingualSrc',
        'tts_model': 'Multilingual_Src',
        'text_source': 'src_text',
        'text_format': 'plain',
        'descriptor': 'src',
    },
]

# Language mappings
LANGUAGES = {
    'efi': 'efik',
    'ibo': 'igbo',
    'swa': 'swahili',
    'xho': 'xhosa'
}


def format_text_for_pipeline(df: pd.DataFrame, pipeline_config: dict, iso_code: str) -> pd.DataFrame:
    """
    Format text column based on pipeline configuration.

    Args:
        df: DataFrame with source_text and predicted_tgt_text columns
        pipeline_config: Pipeline configuration dict
        iso_code: Language ISO code (efi, ibo, swa, xho)

    Returns:
        DataFrame with formatted 'text' column
    """
    df = df.copy()

    text_format = pipeline_config['text_format']
    text_source = pipeline_config['text_source']

    if text_format == 'plain':
        # Just use the specified source directly
        df['text'] = df[text_source]

    elif text_format == 'custom_lang':
        # <eng>{src_text} <{iso}>{predicted_tgt_text}
        df['text'] = df.apply(
            lambda row: f"<eng>{row['src_text']} <{iso_code}>{row['predicted_tgt_text']}",
            axis=1
        )

    elif text_format == 'custom_translate':
        # <translate> <eng>{src_text} <{iso}>{predicted_tgt_text}
        df['text'] = df.apply(
            lambda row: f"<translate> <eng>{row['src_text']} <{iso_code}>{row['predicted_tgt_text']}",
            axis=1
        )

    return df


def generate_formatted_csv(
    pipeline_config: dict,
    iso_code: str,
    selected_samples_path: Path,
    output_dir: Path
) -> Path:
    """
    Generate formatted CSV for a specific pipeline and language.

    Args:
        pipeline_config: Pipeline configuration dict
        iso_code: Language ISO code
        selected_samples_path: Path to selected samples CSV
        output_dir: Output directory for formatted CSV

    Returns:
        Path to generated CSV file
    """
    # Load selected samples
    df = pd.read_csv(selected_samples_path, sep='|')

    # Format text according to pipeline config
    df = format_text_for_pipeline(df, pipeline_config, iso_code)

    # Save formatted CSV
    output_path = output_dir / f"pipeline_{pipeline_config['id']}_{iso_code}_{pipeline_config['name']}.csv"
    df.to_csv(output_path, sep='|', index=False)

    return output_path


def generate_slurm_script(
    pipeline_config: dict,
    iso_code: str,
    formatted_csv_path: Path,
    output_dir: Path,
    tts_venv_path: str,
    log_dir: Path
) -> Path:
    """
    Generate SLURM batch script for a pipeline/language combination.

    Args:
        pipeline_config: Pipeline configuration dict
        iso_code: Language ISO code
        formatted_csv_path: Path to formatted input CSV
        output_dir: Base output directory for synthesis
        tts_venv_path: Path to TTS virtual environment
        log_dir: Directory for SLURM logs

    Returns:
        Path to generated SLURM script
    """
    pipeline_id = pipeline_config['id']
    model_path = TTS_MODELS[pipeline_config['tts_model']]
    descriptor = pipeline_config['descriptor']

    script_content = f"""#!/bin/bash
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=syn_p{pipeline_id}_{iso_code}
#SBATCH --output={log_dir}/pipeline_{pipeline_id}_{iso_code}_%j.out
#SBATCH --error={log_dir}/pipeline_{pipeline_id}_{iso_code}_%j.err

echo "=========================================="
echo "TTS Synthesis: Pipeline {pipeline_id}"
echo "Pipeline: {pipeline_config['name']}"
echo "Language: {LANGUAGES[iso_code]} ({iso_code})"
echo "TTS Model: {pipeline_config['tts_model']}"
echo "Descriptor: {descriptor}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Activate TTS virtual environment
source {tts_venv_path}/bin/activate

# Navigate to TTS directory
cd /home/vacl2/multimodal_translation/services/tts

# Run batch synthesis
PYTHONPATH=$(pwd):$PYTHONPATH uv run python batch_synthesize.py \\
    --csv-file {formatted_csv_path} \\
    --output-dir {output_dir} \\
    --language {iso_code} \\
    --model-path {model_path} \\
    --descriptor {descriptor} \\
    --device auto \\
    --sample-rate 16000 \\
    --max-samples 300 \\
    --text-column text

echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $?"
echo "=========================================="
"""

    # Save script
    script_path = output_dir / f"pipeline_{pipeline_id}_{iso_code}.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM jobs for multi-pipeline TTS synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--submit',
        action='store_true',
        help='Submit jobs to SLURM (default: dry run, generate scripts only)'
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
        '--log-dir',
        type=str,
        default='/home/vacl2/multimodal_translation/services/tts/_logs/multi_pipeline_synthesis',
        help='Directory for SLURM job logs'
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    data_base_dir = Path(args.data_base_dir)
    tts_venv_path = Path(args.tts_venv)
    log_dir = Path(args.log_dir)

    # Create output directories
    formatted_csv_dir = eval_dir / 'formatted_inputs'
    slurm_scripts_dir = eval_dir / 'slurm_jobs'
    formatted_csv_dir.mkdir(parents=True, exist_ok=True)
    slurm_scripts_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("MULTI-PIPELINE TTS SYNTHESIS ORCHESTRATOR")
    logger.info("="*70)
    logger.info(f"Mode: {'SUBMIT JOBS' if args.submit else 'DRY RUN (generate scripts only)'}")
    logger.info(f"Pipelines: {len(PIPELINE_CONFIGS)}")
    logger.info(f"Languages: {len(LANGUAGES)}")
    logger.info(f"Total jobs: {len(PIPELINE_CONFIGS) * len(LANGUAGES)}")
    logger.info("")

    job_ids = {}
    generated_scripts = []

    # Generate formatted CSVs and SLURM scripts for each pipeline/language
    for pipeline_config in PIPELINE_CONFIGS:
        pipeline_id = pipeline_config['id']
        logger.info(f"\nPipeline {pipeline_id}: {pipeline_config['name']}")
        logger.info(f"  TTS Model: {pipeline_config['tts_model']}")
        logger.info(f"  Text Source: {pipeline_config['text_source']}")
        logger.info(f"  Text Format: {pipeline_config['text_format']}")
        logger.info(f"  Descriptor: {pipeline_config['descriptor']}")

        for iso_code in LANGUAGES.keys():
            language_name = LANGUAGES[iso_code]

            # Path to selected samples
            selected_samples_path = eval_dir / f"selected_samples_{iso_code}.csv"

            if not selected_samples_path.exists():
                logger.error(f"  ✗ {language_name}: Selected samples not found: {selected_samples_path}")
                logger.error(f"    Run select_synthesis_samples.py first!")
                continue

            # Generate formatted CSV
            formatted_csv_path = generate_formatted_csv(
                pipeline_config=pipeline_config,
                iso_code=iso_code,
                selected_samples_path=selected_samples_path,
                output_dir=formatted_csv_dir
            )
            logger.info(f"  ✓ {language_name}: Generated formatted CSV: {formatted_csv_path.name}")

            # Generate SLURM script
            slurm_script_path = generate_slurm_script(
                pipeline_config=pipeline_config,
                iso_code=iso_code,
                formatted_csv_path=formatted_csv_path,
                output_dir=data_base_dir,
                tts_venv_path=tts_venv_path,
                log_dir=log_dir
            )
            logger.info(f"  ✓ {language_name}: Generated SLURM script: {slurm_script_path.name}")
            generated_scripts.append(slurm_script_path)

            # Submit job if requested
            if args.submit:
                try:
                    result = subprocess.run(
                        ['sbatch', str(slurm_script_path)],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    # Extract job ID from output (format: "Submitted batch job 12345")
                    job_id = result.stdout.strip().split()[-1]
                    job_key = f"pipeline_{pipeline_id}_{iso_code}"
                    job_ids[job_key] = job_id
                    logger.info(f"  ✓ {language_name}: Submitted job {job_id}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"  ✗ {language_name}: Failed to submit job: {e}")

    # Save job IDs to JSON
    if args.submit and job_ids:
        job_ids_path = slurm_scripts_dir / 'job_ids.json'
        with open(job_ids_path, 'w') as f:
            json.dump(job_ids, f, indent=2)
        logger.info(f"\n✓ Job IDs saved to: {job_ids_path}")

    logger.info("\n" + "="*70)
    logger.info("ORCHESTRATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Generated {len(generated_scripts)} SLURM scripts")

    if args.submit:
        logger.info(f"Submitted {len(job_ids)} jobs")
        logger.info(f"\nMonitor jobs with: squeue -u $USER")
        logger.info(f"Check logs in: {log_dir}")
        logger.info(f"\nSynthesis output will be saved to:")
        logger.info(f"  {data_base_dir}/{{language}}/predicted_{{descriptor}}_{{model_checkpoint}}/")
    else:
        logger.info("\nDRY RUN: Scripts generated but not submitted")
        logger.info(f"To submit jobs, run with --submit flag:")
        logger.info(f"  python {Path(__file__).name} --submit")


if __name__ == '__main__':
    main()
