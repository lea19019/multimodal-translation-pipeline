"""
Quick CPU test of the fine-tuning script
Tests with a tiny batch to verify gradient flow and training loop
"""
import torch
import sys
from pathlib import Path

print("="*60)
print("Testing Fine-tuning Script on CPU")
print("="*60)

# Force CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from finetune_speech_encoder import SpeechEncoderTrainer, SpeechTextDataset, collate_fn
from torch.utils.data import DataLoader

# Test configuration
LANG_CODE = 'xho'  # Test with Swahili
BATCH_SIZE = 2
NUM_BATCHES = 2  # Just test 2 batches

print(f"\nTest Configuration:")
print(f"  Language: {LANG_CODE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Number of batches: {NUM_BATCHES}")

# Step 1: Create trainer
print(f"\n{'='*60}")
print("Step 1: Creating trainer...")
print(f"{'='*60}")

try:
    trainer = SpeechEncoderTrainer(
        lang_code=LANG_CODE,
        learning_rate=1e-5,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=4  # Match the config in main script
    )
    print("✓ Trainer created successfully")
    print(f"  Device: {trainer.device}")
    print(f"  Model in training mode: {trainer.speech_model.training}")
except Exception as e:
    print(f"✗ Failed to create trainer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Load data
print(f"\n{'='*60}")
print("Step 2: Loading dataset...")
print(f"{'='*60}")

try:
    data_dir = Path("/home/vacl2/multimodal_translation/services/data/languages/swahili")
    csv_path = data_dir / "metadata_train.csv"
    
    if not csv_path.exists():
        print(f"✗ CSV not found: {csv_path}")
        sys.exit(1)
    
    dataset = SpeechTextDataset(csv_path)
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Create a tiny dataloader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=0,  # Use 0 workers for CPU testing
        shuffle=True
    )
    print(f"✓ DataLoader created")
    
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test forward pass
print(f"\n{'='*60}")
print("Step 3: Testing forward pass...")
print(f"{'='*60}")

try:
    batch = next(iter(loader))
    print(f"  Audio shape: {batch['audio'].shape}")
    print(f"  Texts: {len(batch['text'])} samples")
    print(f"  Lengths: {batch['lengths'].tolist()}")
    
    # Get embeddings
    print("\n  Getting speech embeddings...")
    print(f"  Min length: {batch['lengths'].min().item()}, Max length: {batch['lengths'].max().item()}")
    speech_emb = trainer.get_speech_embedding(batch['audio'], batch['lengths'])
    print(f"  ✓ Speech embeddings: {speech_emb.shape}, requires_grad={speech_emb.requires_grad}")
    
    print("\n  Getting text embeddings...")
    text_emb = trainer.get_text_embedding(batch['text'])
    print(f"  ✓ Text embeddings: {text_emb.shape}")
    
    print("\n✓ Forward pass successful!")
    
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test training step (with backward pass)
print(f"\n{'='*60}")
print("Step 4: Testing training step with backward pass...")
print(f"{'='*60}")

try:
    # Get a fresh batch
    batch_iter = iter(loader)
    
    for i in range(NUM_BATCHES):
        batch = next(batch_iter)
        print(f"\n  Batch {i+1}/{NUM_BATCHES}")
        
        # Training step (pass step number for gradient accumulation)
        loss = trainer.train_step(batch, accumulation_step=i)
        print(f"    Loss: {loss:.4f}")
        
        # Check that gradients were computed
        has_grad = False
        for param in trainer.speech_model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if has_grad:
            print(f"    ✓ Gradients computed successfully")
        else:
            print(f"    ✗ No gradients found!")
            sys.exit(1)
    
    print("\n✓ Training steps successful!")
    
except StopIteration:
    print(f"✗ Not enough data for {NUM_BATCHES} batches")
    sys.exit(1)
except Exception as e:
    print(f"✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test evaluation step
print(f"\n{'='*60}")
print("Step 5: Testing evaluation step...")
print(f"{'='*60}")

try:
    trainer.speech_model.eval()
    batch = next(iter(loader))
    
    eval_loss = trainer.eval_step(batch)
    print(f"  Eval loss: {eval_loss:.4f}")
    print("✓ Evaluation step successful!")
    
except Exception as e:
    print(f"✗ Evaluation step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print(f"\n{'='*60}")
print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
print(f"{'='*60}")
print("\nThe fine-tuning script is working correctly!")
print("You can now safely submit the job to SLURM:")
print("  sbatch finetune_speech_encoder.sh")
print(f"{'='*60}\n")
