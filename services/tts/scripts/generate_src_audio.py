"""
Generate source audio from text using XTTS model
Processes mapped_metadata_test.csv files for multiple languages
"""

import logging
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from tqdm import tqdm
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = Path("/home/vacl2/multimodal_translation/services/tts/checkpoints/XTTS_v2.0_original_model_files")
REFERENCE_AUDIO = Path("/home/vacl2/multimodal_translation/services/tts/reference_audio/female_en.wav")
BASE_DATA_DIR = Path("/home/vacl2/multimodal_translation/services/data/languages")
LANGUAGES = ["swahili", "xhosa"]#["efik", "igbo", "swahili", "xhosa"]

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_xtts_model(model_dir: Path, device: str = "cuda"):
    """
    Load XTTS model
    
    Args:
        model_dir: Path to model directory
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (config, model)
    """
    logger.info(f"Loading XTTS model from {model_dir} on {device}...")
    
    # Check for required files
    config_path = model_dir / "config.json"
    model_path = model_dir / "model.pth"
    
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Required model files not found in {model_dir}. "
            f"Expected config.json and model.pth"
        )
    
    # Load XTTS model
    config = XttsConfig()
    config.load_json(str(config_path))
    
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True, use_deepspeed=False)
    model.to(device)
    
    logger.info(f"XTTS model loaded successfully on {device}")
    
    return config, model


def get_xtts_language_code(lang_name: str) -> str:
    """
    Convert language name to XTTS-compatible format.
    
    Args:
        lang_name: Language name (e.g., 'efik', 'igbo', 'xhosa', 'swahili')
        
    Returns:
        XTTS-compatible language code
    """
    language_map = {
        'efik': 'en',  # XTTS doesn't have efik, use English as base
        'igbo': 'en',  # XTTS doesn't have igbo, use English as base
        'xhosa': 'en', # XTTS doesn't have xhosa, use English as base
        'swahili': 'en', # XTTS doesn't have swahili, use English as base
        'yoruba': 'en', # XTTS doesn't have yoruba, use English as base
    }
    
    return language_map.get(lang_name.lower(), 'en')


def generate_audio(
    model, 
    text: str, 
    language: str, 
    gpt_cond_latent, 
    speaker_embedding,
    output_path: Path
):
    """
    Generate audio from text and save to file
    
    Args:
        model: XTTS model
        text: Text to synthesize
        language: Language code
        gpt_cond_latent: GPT conditioning latent from reference audio
        speaker_embedding: Speaker embedding from reference audio
        output_path: Path to save audio file
    """
    try:
        # Generate speech
        out = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.7,
            length_penalty=1.0,
            repetition_penalty=5.0,
            top_k=50,
            top_p=0.85,
        )
        
        # Extract waveform
        if isinstance(out, dict):
            wav = out.get("wav")
        else:
            wav = out
        
        if wav is None:
            raise ValueError("Model inference returned None")
        
        # Convert to numpy array if not already
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)
        
        # Convert to float32 format
        wav = wav.astype(np.float32)
        
        # Save audio file (XTTS outputs at 24kHz)
        sf.write(str(output_path), wav, 24000)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate audio for text '{text[:50]}...': {e}")
        return False


def process_language(
    language: str,
    model,
    gpt_cond_latent,
    speaker_embedding,
    base_dir: Path
):
    """
    Process all test data for a language
    
    Args:
        language: Language name (e.g., 'efik')
        model: Loaded XTTS model
        gpt_cond_latent: GPT conditioning latent
        speaker_embedding: Speaker embedding
        base_dir: Base data directory
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {language}...")
    logger.info(f"{'='*60}")
    
    lang_dir = base_dir / language
    csv_path = lang_dir / "mapped_metadata_test.csv"
    output_dir = lang_dir / "src_audio"
    
    # Check if CSV exists
    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Read CSV
    df = pd.read_csv(csv_path, sep='|')
    logger.info(f"Found {len(df)} samples to process")
    
    # Get XTTS language code
    xtts_lang = get_xtts_language_code(language)
    logger.info(f"Using XTTS language code: {xtts_lang}")
    
    # Process each row
    success_count = 0
    failed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating {language} audio"):
        try:
            # Get source text
            src_text = row['src_text']
            
            if pd.isna(src_text) or not str(src_text).strip():
                logger.warning(f"Empty source text at row {idx}, skipping")
                failed_count += 1
                continue
            
            # Create output filename based on segment and user
            segment_id = row['segment_id']
            user_id = row['user_id']
            output_filename = f"Segment={segment_id}_User={user_id}_Language=en_src.wav"
            output_path = output_dir / output_filename
            
            # Skip if already exists
            if output_path.exists():
                logger.debug(f"Audio already exists: {output_filename}, skipping")
                success_count += 1
                continue
            
            # Generate audio
            success = generate_audio(
                model=model,
                text=str(src_text),
                language=xtts_lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                output_path=output_path
            )
            
            if success:
                success_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            failed_count += 1
    
    logger.info(f"\n{language} completed:")
    logger.info(f"  ✓ Success: {success_count}/{len(df)}")
    logger.info(f"  ✗ Failed: {failed_count}/{len(df)}")


def main():
    """Main processing function"""
    logger.info("Starting source audio generation...")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model: {MODEL_DIR}")
    logger.info(f"Reference audio: {REFERENCE_AUDIO}")
    logger.info(f"Languages: {LANGUAGES}")
    
    # Check reference audio exists
    if not REFERENCE_AUDIO.exists():
        logger.error(f"Reference audio not found: {REFERENCE_AUDIO}")
        return
    
    # Load model once
    config, model = load_xtts_model(MODEL_DIR, DEVICE)
    
    # Compute speaker latents once (same reference for all)
    logger.info("Computing speaker latents from reference audio...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=str(REFERENCE_AUDIO)
    )
    logger.info("Speaker latents computed")
    
    # Process each language
    for language in LANGUAGES:
        process_language(
            language=language,
            model=model,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            base_dir=BASE_DATA_DIR
        )
    
    logger.info("\n" + "="*60)
    logger.info("All languages processed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()