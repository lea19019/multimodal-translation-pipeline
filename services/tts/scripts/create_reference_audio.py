"""
Script to create a default reference audio file for XTTS voice cloning.
This will download or create a simple reference voice sample.
"""

import os
import urllib.request
from pathlib import Path

def download_reference_audio():
    """Download a sample reference audio file"""
    
    # Create reference audio directory
    ref_dir = Path(__file__).parent / "reference_audio"
    ref_dir.mkdir(exist_ok=True)
    
    # Sample audio URLs (royalty-free samples for TTS)
    # These are short audio samples that can be used for voice cloning
    samples = {
        "female_en.wav": "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0001.wav",
        # Add more if needed
    }
    
    print("Downloading reference audio samples...")
    
    for filename, url in samples.items():
        output_path = ref_dir / filename
        if output_path.exists():
            print(f"  ✓ {filename} already exists")
            continue
            
        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, output_path)
            print(f"  ✓ Downloaded {filename}")
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
    
    print(f"\nReference audio files saved to: {ref_dir}")
    print(f"Default reference: {ref_dir / 'female_en.wav'}")
    
    return ref_dir / "female_en.wav"

if __name__ == "__main__":
    ref_path = download_reference_audio()
    if ref_path.exists():
        print(f"\n✓ Success! Use this path in your TTS service: {ref_path}")
    else:
        print("\n✗ Failed to create reference audio")
