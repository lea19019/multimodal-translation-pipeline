#!/usr/bin/env python3
"""
Batch NMT Prediction Script for Multimodal Translation Evaluation

This script performs GPU-accelerated batch inference using the fine-tuned NLLB model
to generate translation predictions for evaluation purposes.

Usage:
    python batch_predict.py --languages efik igbo swahili xhosa --batch-size 64

Output:
    Creates nmt_predictions.csv in each language folder with columns:
    segment_id|user_id|speaker_id|src_text|predicted_tgt_text|ground_truth_tgt_text|iso_code
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Language code mapping: ISO 3-letter codes to NLLB format
ISO_TO_NLLB = {
    'efi': 'efi_Latn',  # Efik
    'ibo': 'ibo_Latn',  # Igbo
    'swh': 'swh_Latn',  # Swahili
    'xho': 'xho_Latn',  # Xhosa
}

LANGUAGE_NAME_MAPPING = {
    'efik': 'efi',
    'igbo': 'ibo',
    'swahili': 'swh',
    'xhosa': 'xho',
}


class NMTBatchPredictor:
    """Handles batch NMT prediction for multiple languages."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        batch_size: int = 64,
    ):
        """
        Initialize the batch predictor.

        Args:
            model_path: Path to the fine-tuned NLLB model checkpoint
            device: Device to use ("cuda", "cpu", or "auto")
            batch_size: Number of samples to process at once
        """
        self.model_path = Path(model_path)
        self.batch_size = batch_size

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        logger.info(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path))
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")

    def predict_batch(
        self,
        texts: List[str],
        target_lang: str,
        max_length: int = 128,
        num_beams: int = 5,
    ) -> List[str]:
        """
        Generate predictions for a batch of texts.

        Args:
            texts: List of source texts to translate
            target_lang: Target language code (NLLB format, e.g., 'efi_Latn')
            max_length: Maximum length of generated sequences
            num_beams: Number of beams for beam search (1 = greedy)

        Returns:
            List of translated texts
        """
        # Set source language
        self.tokenizer.src_lang = "eng_Latn"

        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # Get target language token ID
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_lang)

        # Generate translations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=num_beams,
            )

        # Decode outputs
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return decoded

    def process_language(
        self,
        language: str,
        data_dir: Path,
        model_name: str,
    ) -> Dict:
        """
        Process all test samples for a single language.

        Args:
            language: Language name (efik, igbo, swahili, xhosa)
            data_dir: Base data directory containing language folders
            model_name: Model name to include in output filename

        Returns:
            Dictionary with processing statistics
        """
        # Map language name to ISO code
        iso_code = LANGUAGE_NAME_MAPPING.get(language.lower())
        if not iso_code:
            raise ValueError(f"Unknown language: {language}")

        # Get NLLB language code
        target_lang = ISO_TO_NLLB[iso_code]

        # Paths
        lang_dir = data_dir / language
        input_csv = lang_dir / "mapped_metadata_test.csv"
        output_csv = lang_dir / f"nmt_predictions_{model_name}.csv"

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {language.upper()} ({iso_code} -> {target_lang})")
        logger.info(f"{'='*60}")
        logger.info(f"Input: {input_csv}")
        logger.info(f"Output: {output_csv}")

        # Check if input exists
        if not input_csv.exists():
            logger.error(f"Input CSV not found: {input_csv}")
            return {"language": language, "status": "error", "samples": 0}

        # Load test data
        logger.info("Loading test data...")
        df = pd.read_csv(input_csv, sep="|")
        total_samples = len(df)
        logger.info(f"Loaded {total_samples} samples")

        # Prepare for batch processing
        predictions = []
        errors = []

        # Process in batches
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size

        logger.info(f"Processing in {num_batches} batches (batch_size={self.batch_size})")

        with tqdm(total=total_samples, desc=f"{language}", unit="sample") as pbar:
            for i in range(0, total_samples, self.batch_size):
                batch_df = df.iloc[i:i+self.batch_size]
                batch_texts = batch_df['src_text'].tolist()

                try:
                    # Generate predictions
                    batch_predictions = self.predict_batch(
                        batch_texts,
                        target_lang=target_lang,
                        max_length=128,
                        num_beams=5,
                    )

                    predictions.extend(batch_predictions)

                    # Clear GPU cache periodically
                    if self.device == "cuda" and i % (self.batch_size * 10) == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Error processing batch {i//self.batch_size}: {e}")
                    # Add empty predictions for failed batch
                    predictions.extend([""] * len(batch_texts))
                    errors.append({
                        'batch': i//self.batch_size,
                        'error': str(e)
                    })

                pbar.update(len(batch_texts))

        # Add predictions to dataframe
        df['predicted_tgt_text'] = predictions
        df['ground_truth_tgt_text'] = df['tgt_text']

        # Reorder columns for output
        output_columns = [
            'segment_id',
            'user_id',
            'speaker_id',
            'src_text',
            'predicted_tgt_text',
            'ground_truth_tgt_text',
            'iso_code'
        ]

        df_output = df[output_columns]

        # Save predictions
        logger.info(f"Saving predictions to {output_csv}")
        df_output.to_csv(output_csv, sep="|", index=False)

        # Calculate statistics
        non_empty = sum(1 for p in predictions if p.strip())

        logger.info(f"Completed {language}:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Successful predictions: {non_empty}")
        logger.info(f"  Failed predictions: {total_samples - non_empty}")
        logger.info(f"  Errors: {len(errors)}")

        return {
            'language': language,
            'status': 'success',
            'total_samples': total_samples,
            'successful': non_empty,
            'failed': total_samples - non_empty,
            'errors': len(errors),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Batch NMT prediction for multimodal translation evaluation"
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['efik', 'igbo', 'swahili', 'xhosa'],
        choices=['efik', 'igbo', 'swahili', 'xhosa'],
        help='Languages to process (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/home/vacl2/multimodal_translation/services/data/languages',
        help='Base directory containing language folders'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='/home/vacl2/multimodal_translation/services/nmt/checkpoints/multilang_finetuned_final',
        help='Path to fine-tuned NLLB model'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name for output files (auto-extracted from model-path if not provided)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for inference (default: 64)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--num-beams',
        type=int,
        default=5,
        help='Number of beams for beam search (default: 5, use 1 for greedy)'
    )

    args = parser.parse_args()

    # Auto-extract model name from path if not provided
    if not args.model_name:
        args.model_name = Path(args.model_path).name

    logger.info(f"Using model: {args.model_name}")
    logger.info(f"Model path: {args.model_path}")

    try:
        # Initialize predictor
        predictor = NMTBatchPredictor(
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
        )

        # Process each language
        data_dir = Path(args.data_dir)
        results = []

        for language in args.languages:
            result = predictor.process_language(language, data_dir, args.model_name)
            results.append(result)

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("BATCH PREDICTION COMPLETE")
        logger.info(f"{'='*60}")

        total_samples = sum(r['total_samples'] for r in results if r['status'] == 'success')
        total_successful = sum(r['successful'] for r in results if r['status'] == 'success')

        logger.info(f"Total samples processed: {total_samples}")
        logger.info(f"Successful predictions: {total_successful}")
        logger.info(f"Success rate: {total_successful/total_samples*100:.2f}%")

        logger.info("\nPer-language results:")
        for result in results:
            if result['status'] == 'success':
                logger.info(f"  {result['language']:10s}: {result['successful']:5d}/{result['total_samples']:5d} ({result['successful']/result['total_samples']*100:.1f}%)")
            else:
                logger.info(f"  {result['language']:10s}: ERROR")

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
