#!/usr/bin/env python3
"""
Resample Audio Samples to 16kHz

This script resamples all audio files in the evaluation data directories to 16kHz,
which is required by SONAR speech encoders used in BLASER evaluation.

Usage:
    python scripts/resample_audio_samples.py --data-dir ../data/app_evaluation
    python scripts/resample_audio_samples.py --data-dir ../data/app_evaluation --dry-run
"""

import argparse
import logging
from pathlib import Path
from typing import List

import soundfile as sf
import torch
from torchaudio.transforms import Resample

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def find_audio_files(data_dir: Path) -> List[Path]:
    """
    Find all audio files in the data directory.

    Args:
        data_dir: Root data directory

    Returns:
        List of audio file paths
    """
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(data_dir.rglob(f'*{ext}'))

    return sorted(audio_files)


def resample_audio_file(audio_path: Path, target_sr: int = TARGET_SAMPLE_RATE, dry_run: bool = False) -> bool:
    """
    Resample a single audio file to target sample rate.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000)
        dry_run: If True, don't actually modify files

    Returns:
        True if file was resampled, False if already at target rate
    """
    try:
        # Load audio using soundfile
        waveform, sample_rate = sf.read(str(audio_path))

        # Check if resampling is needed
        if sample_rate == target_sr:
            logger.debug(f"✓ {audio_path.name} already at {target_sr}Hz")
            return False

        logger.info(f"Resampling {audio_path.name}: {sample_rate}Hz → {target_sr}Hz")

        if dry_run:
            logger.info(f"  [DRY RUN] Would resample {audio_path}")
            return True

        # Convert to torch tensor for resampling
        waveform_tensor = torch.from_numpy(waveform).float()

        # Handle mono vs stereo
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)  # Add channel dimension
        else:
            waveform_tensor = waveform_tensor.T  # Transpose to (channels, samples)

        # Resample
        resampler = Resample(
            orig_freq=sample_rate,
            new_freq=target_sr
        )
        resampled_waveform = resampler(waveform_tensor)

        # Convert back to numpy and transpose if needed
        resampled_numpy = resampled_waveform.numpy()
        if resampled_numpy.shape[0] == 1:
            resampled_numpy = resampled_numpy.squeeze(0)  # Remove channel dimension for mono
        else:
            resampled_numpy = resampled_numpy.T  # Transpose back to (samples, channels)

        # Backup original file
        backup_path = audio_path.with_suffix(f'.{sample_rate}hz.backup')
        if not backup_path.exists():
            audio_path.rename(backup_path)
            logger.debug(f"  Backed up to {backup_path.name}")

        # Save resampled audio
        sf.write(str(audio_path), resampled_numpy, target_sr)
        logger.info(f"  ✓ Saved resampled audio to {audio_path.name}")

        return True

    except Exception as e:
        logger.error(f"Failed to resample {audio_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Resample audio samples to 16kHz for BLASER evaluation'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('../data/app_evaluation'),
        help='Root data directory containing audio samples'
    )
    parser.add_argument(
        '--target-rate',
        type=int,
        default=TARGET_SAMPLE_RATE,
        help='Target sample rate in Hz (default: 16000)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without modifying files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return 1

    logger.info(f"Scanning for audio files in: {args.data_dir}")
    audio_files = find_audio_files(args.data_dir)

    if not audio_files:
        logger.warning("No audio files found")
        return 0

    logger.info(f"Found {len(audio_files)} audio files")

    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")

    # Resample files
    resampled_count = 0
    skipped_count = 0
    error_count = 0

    for audio_file in audio_files:
        try:
            was_resampled = resample_audio_file(
                audio_file,
                target_sr=args.target_rate,
                dry_run=args.dry_run
            )
            if was_resampled:
                resampled_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            error_count += 1

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESAMPLING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files found:     {len(audio_files)}")
    logger.info(f"Files resampled:       {resampled_count}")
    logger.info(f"Files already OK:      {skipped_count}")
    logger.info(f"Errors:                {error_count}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("\nThis was a DRY RUN. Run without --dry-run to apply changes.")
    else:
        logger.info(f"\nOriginal files backed up with '.{args.target_rate}hz.backup' extension")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    exit(main())
