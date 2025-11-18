"""
Download and load SONAR speech encoders for fine-tuning
"""

import torch
import os
import shutil
from pathlib import Path
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

DEVICE = torch.device("cpu")
CHECKPOINT_DIR = Path("./checkpoints")

def get_default_cache():
    """Get fairseq2 default cache location"""
    torch_home = os.environ.get('TORCH_HOME', Path.home() / '.cache' / 'torch')
    return Path(torch_home) / 'hub'

def save_encoder_weights(encoder, lang):
    """Save encoder model weights"""
    save_path = CHECKPOINT_DIR / f"encoder_{lang}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save the actual model state dict
    model_file = save_path / "model.pt"
    torch.save(encoder.model.state_dict(), model_file)
    print(f"  Saved weights to {model_file}")
    
    return save_path

def load_encoder_weights(encoder_name, lang):
    """Load encoder weights if they exist"""
    weight_file = CHECKPOINT_DIR / f"encoder_{lang}" / "model.pt"
    
    if weight_file.exists():
        print(f"Loading cached {encoder_name} from {weight_file}")
        # Load base encoder first
        encoder = SpeechToEmbeddingModelPipeline(
            encoder=encoder_name,
            device=DEVICE
        )
        # Load saved weights
        encoder.model.load_state_dict(torch.load(weight_file, map_location=DEVICE))
        return encoder
    
    return None

def download_encoder(encoder_name, lang):
    """Download encoder and save weights"""
    print(f"Downloading {encoder_name}...")
    
    encoder = SpeechToEmbeddingModelPipeline(
        encoder=encoder_name,
        device=DEVICE
    )
    
    save_encoder_weights(encoder, lang)
    print(f"✓ {lang} downloaded and saved")
    
    return encoder

def load_encoders():
    """Load or download encoders"""
    
    encoders_to_load = {
        'swh': 'sonar_speech_encoder_swh',
        'eng': 'sonar_speech_encoder_eng'
    }
    
    loaded = {}
    
    for lang, encoder_name in encoders_to_load.items():
        # Try loading from checkpoint
        encoder = load_encoder_weights(encoder_name, lang)
        
        # Download if not found
        if encoder is None:
            encoder = download_encoder(encoder_name, lang)
        
        loaded[lang] = encoder
    
    return loaded

if __name__ == "__main__":
    print("Loading encoders...")
    encoders = load_encoders()
    
    # Test
    dummy = torch.zeros(1, 16000)
    print("\nTesting encoders:")
    for lang, enc in encoders.items():
        emb = enc.predict([dummy])
        print(f"{lang}: {emb.shape}")
    
    print("\nDone. Use encoders['swh'] or encoders['eng']")