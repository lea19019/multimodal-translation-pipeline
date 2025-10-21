#!/usr/bin/env python3
"""
Verify Model Cache

This script verifies that all BLASER and SONAR models are properly cached
in the local models/ directory and that environment variables are correctly set.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Expected model files and their approximate sizes (in bytes)
EXPECTED_FILES = {
    # BLASER models
    "8044776ae0d161be55abc16a/model.pt": (60_000_000, 70_000_000),  # blaser_2_0_qe: ~67 MB
    "23dd0ae040ead1082271a669/model.pt": (85_000_000, 95_000_000),  # blaser_2_0_ref: ~91 MB

    # Text encoder
    "43e937cce6a6a5cd5730cf66/sonar_text_encoder.pt": (2_700_000_000, 3_100_000_000),  # ~2.9 GB
    "538cc2c5593d2eec13df5a8f/sentencepiece.source.256000.model": (4_000_000, 5_000_000),  # ~4.7 MB

    # Speech encoders
    "5f4057a558a791ac037a743c/spenc.eng.pt": (7_000_000_000, 8_000_000_000),  # English: ~7.5 GB
    "ab3bc75032cee834ffe8bad3/spenc.v3ap.spa.pt": (7_500_000_000, 8_700_000_000),  # Spanish: ~8.1 GB
}

MODEL_NAMES = {
    "8044776ae0d161be55abc16a/model.pt": "BLASER 2.0 QE",
    "23dd0ae040ead1082271a669/model.pt": "BLASER 2.0 Reference",
    "43e937cce6a6a5cd5730cf66/sonar_text_encoder.pt": "SONAR Text Encoder",
    "538cc2c5593d2eec13df5a8f/sentencepiece.source.256000.model": "SONAR Tokenizer",
    "5f4057a558a791ac037a743c/spenc.eng.pt": "SONAR English Speech Encoder",
    "ab3bc75032cee834ffe8bad3/spenc.v3ap.spa.pt": "SONAR Spanish Speech Encoder",
}


def format_size(bytes_size):
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def verify_environment():
    """Verify environment variables are set."""
    print("=" * 70)
    print("CHECKING ENVIRONMENT VARIABLES")
    print("=" * 70)

    required_vars = {
        'FAIRSEQ2_CACHE_DIR': Path(__file__).parent.parent / "models",
        'HF_HOME': Path(__file__).parent.parent / "models" / "huggingface",
        'TORCH_HOME': Path(__file__).parent.parent / "models" / "torch",
    }

    all_set = True
    for var_name, expected_path in required_vars.items():
        actual_value = os.environ.get(var_name)
        if actual_value:
            print(f"✓ {var_name}: {actual_value}")
            if Path(actual_value) != expected_path:
                print(f"  ⚠ Warning: Expected {expected_path}")
        else:
            print(f"✗ {var_name}: NOT SET")
            print(f"  Expected: {expected_path}")
            all_set = False

    print()
    return all_set


def verify_cache_files():
    """Verify all model files exist in cache."""
    print("=" * 70)
    print("CHECKING CACHED MODEL FILES")
    print("=" * 70)

    models_dir = Path(__file__).parent.parent / "models"
    print(f"Models directory: {models_dir}")
    print()

    if not models_dir.exists():
        print(f"✗ Models directory does not exist: {models_dir}")
        return False

    all_exist = True
    total_size = 0

    for file_path, (min_size, max_size) in EXPECTED_FILES.items():
        full_path = models_dir / file_path
        model_name = MODEL_NAMES.get(file_path, file_path)

        if full_path.exists():
            size = full_path.stat().st_size
            total_size += size

            if min_size <= size <= max_size:
                print(f"✓ {model_name}")
                print(f"  Path: {file_path}")
                print(f"  Size: {format_size(size)}")
            else:
                print(f"⚠ {model_name} (unexpected size)")
                print(f"  Path: {file_path}")
                print(f"  Size: {format_size(size)} (expected {format_size(min_size)}-{format_size(max_size)})")
                all_exist = False
        else:
            print(f"✗ {model_name} - NOT FOUND")
            print(f"  Expected at: {file_path}")
            all_exist = False

        print()

    print("=" * 70)
    print(f"Total cache size: {format_size(total_size)}")
    print("=" * 70)
    print()

    return all_exist


def test_model_loading():
    """Test that models can be loaded from cache."""
    print("=" * 70)
    print("TESTING MODEL LOADING")
    print("=" * 70)

    try:
        # Test BLASER model loading
        print("Loading BLASER model...")
        from sonar.models.blaser.loader import load_blaser_model

        blaser = load_blaser_model("blaser_2_0_qe")
        print("✓ BLASER model loaded successfully")
        del blaser

        # Test text encoder loading
        print("\nLoading SONAR text encoder...")
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

        text_encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
        )
        print("✓ Text encoder loaded successfully")

        # Test text embedding
        test_emb = text_encoder.predict(["Hello world"], source_lang="eng_Latn")
        print(f"✓ Text encoding test successful (shape: {test_emb.shape})")
        del text_encoder

        # Test speech encoder loading
        print("\nLoading SONAR speech encoder...")
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

        speech_encoder = SpeechToEmbeddingModelPipeline(
            encoder="sonar_speech_encoder_eng",
        )
        print("✓ Speech encoder loaded successfully")
        del speech_encoder

        print("\n✓ All models loaded successfully from cache!")
        return True

    except Exception as e:
        print(f"\n✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "BLASER MODEL CACHE VERIFICATION" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Check environment
    env_ok = verify_environment()

    # Check files
    files_ok = verify_cache_files()

    # Test loading
    loading_ok = test_model_loading()

    # Summary
    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    if env_ok and files_ok and loading_ok:
        print("✓ ALL CHECKS PASSED")
        print()
        print("Model cache is properly configured and all models are available.")
        print("BLASER evaluation will use cached models without downloading.")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print()
        if not env_ok:
            print("  - Environment variables not properly set")
        if not files_ok:
            print("  - Some model files missing or corrupted")
        if not loading_ok:
            print("  - Model loading test failed")
        print()
        print("Run download_models.py to download missing models:")
        print("  uv run python scripts/download_models.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
