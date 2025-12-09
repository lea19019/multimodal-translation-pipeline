# Translation-Integrated XTTS (TI-XTTS)

A unified text-to-speech translation model that integrates semantic translation capabilities into XTTS for end-to-end multilingual speech synthesis with voice cloning.

## Overview

This project creates a **single integrated model** that performs:
- Translation (English → Target language)
- Voice cloning
- High-quality speech synthesis

Unlike pipeline approaches that run separate translation and TTS models, TI-XTTS combines them into one architecture with shared gradients and end-to-end optimization.

## Architecture
```
Input: English text + Speaker reference audio
                    ↓
        ┌───────────────────────┐
        │  Semantic Encoder     │  ← NLLB encoder
        │  (Cross-lingual       │
        │   understanding)      │
        └───────────┬───────────┘
                    ↓
          Semantic vectors
          (language-agnostic)
                    ↓
        ┌───────────────────────┐
        │  Translation Decoder  │  ← NLLB decoder
        │  (Target language     │
        │   generation)         │
        └───────────┬───────────┘
                    ↓
        ┌───────────────────────┐
        │  Projection Layer     │  ← Trainable bridge
        │  (NLLB → GPT-2 space) │
        └───────────┬───────────┘
                    ↓
     Target language embeddings
                    │
                    ├─────────────────┐
                    ↓                 ↓
        ┌───────────────────┐  ┌──────────────┐
        │ Speaker Encoder   │  │  GPT-2       │  ← XTTS components
        │ (Perceiver)       │  │  (Audio      │
        │                   │  │   tokens)    │
        └─────────┬─────────┘  └──────┬───────┘
                  │                   │
                  └────────┬──────────┘
                           ↓
                  ┌─────────────────┐
                  │   HiFiGAN       │  ← XTTS vocoder
                  │   (Waveform)    │
                  └────────┬────────┘
                           ↓
              Target language audio
              in speakers voice
```

## Key Differences from Pipeline Approaches

| Aspect | Pipeline (2 models) | TI-XTTS (1 model) |
|--------|-------------------|-------------------|
| **Inference** | 2 separate calls | 1 forward pass |
| **Optimization** | Independent | End-to-end |
| **Memory** | Load 2 models | Load 1 model |
| **Gradient flow** | Blocked between models | Flows through entire architecture |
| **Latency** | Higher (sequential) | Lower (parallel processing) |
| **Flexibility** | Can swap components | Optimized as unit |

## Requirements
```bash
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
TTS>=0.22.0  # Coqui TTS
numpy>=1.24.0
scipy>=1.10.0

# For training
tensorboard>=2.13.0
wandb>=0.15.0  # optional
```

## Installation
```bash
# Clone the repository
git clone <your-repo>
cd translation-xtts

# Install dependencies
pip install -r requirements.txt

# Download pretrained models (if not already in your repo)
python scripts/download_models.py
```

## Model Components

### 1. NLLB-600M (Translation)
- **Encoder**: Semantic understanding of source language
- **Decoder**: Target language generation
- **Languages**: Supports 200+ languages
- **Size**: ~600M parameters

### 2. XTTS (Voice Synthesis)
- **GPT-2 Encoder**: Audio token prediction
- **Perceiver**: Speaker voice conditioning
- **HiFiGAN**: Neural vocoder
- **Size**: ~500M parameters

### 3. Projection Layer (NEW)
- **Purpose**: Bridge NLLB decoder output → GPT-2 input
- **Architecture**: Linear + LayerNorm + Dropout
- **Size**: ~10M parameters

**Total model size**: ~1.1B parameters

## Implementation

### Core Model Architecture
```python
# ti_xtts/model.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.tts.layers.xtts.gpt import GPT
from TTS.tts.layers.xtts.perceiver_encoder import PerceiverResampler
from TTS.tts.layers.xtts.hifigan_decoder import HifiganGenerator
from TTS.tts.layers.xtts.dvae import DiscreteVAE


class TranslationIntegratedXTTS(nn.Module):
    """
    Unified translation + TTS model
    
    Architecture:
    1. NLLB encoder/decoder for translation
    2. Projection layer to bridge NLLB → XTTS
    3. XTTS components for voice-cloned TTS
    """
    
    def __init__(
        self,
        nllb_model_name="facebook/nllb-200-distilled-600M",
        xtts_config_path="checkpoints/base/config.json",
        xtts_checkpoint_path="checkpoints/base/model.pth",
        dvae_checkpoint_path="checkpoints/base/dvae.pth",
        mel_norm_path="checkpoints/base/mel_stats.pth",
        target_languages=["swh_Latn", "xho_Latn", "efi_Latn", "ibo_Latn"],
    ):
        super().__init__()
        
        # ==================== Translation Components ====================
        
        print("Loading NLLB model...")
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)
        
        # Extract encoder and decoder for separate control
        self.semantic_encoder = self.nllb_model.get_encoder()
        self.translation_decoder = self.nllb_model.get_decoder()
        
        # NLLB hidden size (typically 1024 for 600M model)
        self.nllb_hidden_size = self.nllb_model.config.d_model
        
        # ==================== XTTS Components ====================
        
        print("Loading XTTS model...")
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        
        # Load XTTS config
        xtts_config = XttsConfig()
        xtts_config.load_json(xtts_config_path)
        
        # Initialize XTTS
        xtts = Xtts.init_from_config(xtts_config)
        xtts.load_checkpoint(
            xtts_config,
            checkpoint_path=xtts_checkpoint_path,
            vocab_path=xtts_config_path.replace("config.json", "vocab.json"),
            eval=False
        )
        
        # Extract XTTS components
        self.gpt = xtts.gpt
        self.perceiver = xtts.conditioning_encoder
        self.hifigan = xtts.hifigan_decoder
        self.dvae = xtts.dvae
        
        # GPT-2 hidden size (typically 1024)
        self.gpt_hidden_size = xtts_config.gpt_n_model_channels
        
        # ==================== Bridge Components ====================
        
        # Projection layer: NLLB decoder output → GPT-2 input
        self.projection = nn.Sequential(
            nn.Linear(self.nllb_hidden_size, self.gpt_hidden_size),
            nn.LayerNorm(self.gpt_hidden_size),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(self.gpt_hidden_size, self.gpt_hidden_size),
        )
        
        # Target language settings
        self.target_languages = target_languages
        self.src_lang = "eng_Latn"  # English
        
        # Training flags
        self.freeze_nllb = True
        self.freeze_xtts = True
        
        print(f"Model initialized:")
        print(f"  - NLLB hidden size: {self.nllb_hidden_size}")
        print(f"  - GPT-2 hidden size: {self.gpt_hidden_size}")
        print(f"  - Target languages: {target_languages}")
    
    def set_training_stage(self, stage):
        """
        Configure which components are trainable
        
        Args:
            stage: 'projection' | 'translation' | 'end_to_end'
        """
        if stage == "projection":
            # Stage 1: Train only projection layer
            self._freeze_model(self.semantic_encoder)
            self._freeze_model(self.translation_decoder)
            self._freeze_model(self.gpt)
            self._freeze_model(self.hifigan)
            self._unfreeze_model(self.projection)
            print("Training stage: PROJECTION ONLY")
            
        elif stage == "translation":
            # Stage 2: Train NLLB + projection
            self._unfreeze_model(self.semantic_encoder)
            self._unfreeze_model(self.translation_decoder)
            self._unfreeze_model(self.projection)
            self._freeze_model(self.gpt)
            self._freeze_model(self.hifigan)
            print("Training stage: TRANSLATION + PROJECTION")
            
        elif stage == "end_to_end":
            # Stage 3: Train everything except HiFiGAN
            self._unfreeze_model(self.semantic_encoder)
            self._unfreeze_model(self.translation_decoder)
            self._unfreeze_model(self.projection)
            self._unfreeze_model(self.gpt)
            self._freeze_model(self.hifigan)  # Keep vocoder frozen
            print("Training stage: END-TO-END")
        
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def _freeze_model(self, model):
        """Freeze all parameters in a model"""
        for param in model.parameters():
            param.requires_grad = False
    
    def _unfreeze_model(self, model):
        """Unfreeze all parameters in a model"""
        for param in model.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        source_text_ids,
        source_attention_mask,
        speaker_wav,
        target_audio_codes=None,
        target_lang="swh_Latn",
    ):
        """
        Forward pass for training
        
        Args:
            source_text_ids: Tokenized English text (batch_size, src_len)
            source_attention_mask: Attention mask for source (batch_size, src_len)
            speaker_wav: Mel-spectrogram of speaker reference (batch_size, n_mels, time)
            target_audio_codes: Ground truth audio tokens from DVAE (batch_size, audio_len)
            target_lang: Target language code (e.g., 'swh_Latn' for Swahili)
        
        Returns:
            audio_logits: Predicted audio token logits (batch_size, audio_len, vocab_size)
        """
        batch_size = source_text_ids.size(0)
        device = source_text_ids.device
        
        # ==================== Step 1: Semantic Encoding ====================
        
        # Encode English text into semantic representations
        encoder_outputs = self.semantic_encoder(
            input_ids=source_text_ids,
            attention_mask=source_attention_mask,
            return_dict=True,
        )
        
        # Shape: (batch_size, src_len, nllb_hidden_size)
        semantic_vectors = encoder_outputs.last_hidden_state
        
        # ==================== Step 2: Translation Decoding ====================
        
        # Generate target language representations
        # We use teacher forcing during training (if target text is available)
        # For now, well generate autoregressively
        
        # Prepare decoder input (starts with target language token)
        decoder_start_token_id = self.nllb_model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.nllb_tokenizer.lang_code_to_id[target_lang]
        
        # For training, we'll use the encoder outputs directly
        # The decoder will generate target language hidden states
        decoder_outputs = self.translation_decoder(
            encoder_hidden_states=semantic_vectors,
            encoder_attention_mask=source_attention_mask,
            return_dict=True,
        )
        
        # Shape: (batch_size, tgt_len, nllb_hidden_size)
        translation_hidden = decoder_outputs.last_hidden_state
        
        # ==================== Step 3: Project to GPT-2 Space ====================
        
        # Bridge NLLB output to XTTS input format
        # Shape: (batch_size, tgt_len, gpt_hidden_size)
        projected_embeddings = self.projection(translation_hidden)
        
        # ==================== Step 4: Speaker Conditioning ====================
        
        # Extract speaker characteristics from reference audio
        # Shape: (batch_size, 32, gpt_hidden_size)
        speaker_embeddings = self.perceiver(speaker_wav)
        
        # ==================== Step 5: Combine Text + Speaker ====================
        
        # Concatenate speaker embeddings with projected text
        # This is how XTTS conditions on both text and speaker
        # Shape: (batch_size, 32 + tgt_len, gpt_hidden_size)
        conditioned_input = torch.cat([speaker_embeddings, projected_embeddings], dim=1)
        
        # Create attention mask for combined input
        text_mask = torch.ones(
            batch_size, projected_embeddings.size(1),
            dtype=torch.bool, device=device
        )
        combined_mask = torch.cat([
            torch.ones(batch_size, 32, dtype=torch.bool, device=device),
            text_mask
        ], dim=1)
        
        # ==================== Step 6: Generate Audio Tokens ====================
        
        # GPT-2 predicts audio tokens autoregressively
        # During training, we use teacher forcing with target_audio_codes
        
        if target_audio_codes is not None:
            # Training mode: teacher forcing
            audio_logits = self.gpt(
                cond_latents=conditioned_input,
                text_inputs=None,  # We're using cond_latents directly
                mel_inputs=target_audio_codes[:, :-1],  # Shifted right
                return_attentions=False,
                return_latent=False,
            )
            
            return audio_logits
        else:
            # Inference mode: autoregressive generation
            # This is handled by the inference() method
            raise ValueError("Use inference() method for generation without target codes")
    
    @torch.no_grad()
    def inference(
        self,
        source_text,
        speaker_wav_path,
        target_lang="swh_Latn",
        temperature=0.85,
        top_k=50,
        top_p=0.95,
        max_audio_length=500,
    ):
        """
        Generate translated speech
        
        Args:
            source_text: English text string
            speaker_wav_path: Path to speaker reference audio
            target_lang: Target language code (e.g., 'swh_Latn')
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            max_audio_length: Maximum audio token length
        
        Returns:
            audio: Generated waveform (numpy array)
        """
        self.eval()
        device = next(self.parameters()).device
        
        # ==================== Step 1: Tokenize Source Text ====================
        
        # Add source language prefix
        self.nllb_tokenizer.src_lang = self.src_lang
        
        inputs = self.nllb_tokenizer(
            source_text,
            return_tensors="pt",
            padding=True,
            max_length=512,
        ).to(device)
        
        source_ids = inputs.input_ids
        source_mask = inputs.attention_mask
        
        # ==================== Step 2: Semantic Encoding ====================
        
        encoder_outputs = self.semantic_encoder(
            input_ids=source_ids,
            attention_mask=source_mask,
            return_dict=True,
        )
        
        semantic_vectors = encoder_outputs.last_hidden_state
        
        # ==================== Step 3: Translation ====================
        
        # Generate target language token IDs
        # Force target language
        forced_bos_token_id = self.nllb_tokenizer.lang_code_to_id[target_lang]
        
        translated_ids = self.nllb_model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
            num_beams=5,
            early_stopping=True,
        )
        
        # Get hidden states from decoder
        decoder_outputs = self.translation_decoder(
            encoder_hidden_states=semantic_vectors,
            encoder_attention_mask=source_mask,
            decoder_input_ids=translated_ids,
            return_dict=True,
        )
        
        translation_hidden = decoder_outputs.last_hidden_state
        
        # ==================== Step 4: Project to GPT-2 Space ====================
        
        projected_embeddings = self.projection(translation_hidden)
        
        # ==================== Step 5: Load and Process Speaker Audio ====================
        
        import torchaudio
        from TTS.tts.layers.xtts.audio_utils import wav_to_mel_clipping
        
        # Load speaker audio
        speaker_wav, sr = torchaudio.load(speaker_wav_path)
        
        # Resample if needed (XTTS expects 22.05kHz)
        if sr != 22050:
            resampler = torchaudio.transforms.Resample(sr, 22050)
            speaker_wav = resampler(speaker_wav)
        
        # Convert to mel-spectrogram
        speaker_mel = wav_to_mel_clipping(
            speaker_wav.squeeze(0).numpy(),
            mel_norms_file=None,  # Use default
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            power=1,
            normalized=False,
        )
        
        speaker_mel = torch.FloatTensor(speaker_mel).unsqueeze(0).to(device)
        
        # Get speaker embeddings
        speaker_embeddings = self.perceiver(speaker_mel)
        
        # ==================== Step 6: Combine and Generate Audio Tokens ====================
        
        # Concatenate speaker + text
        conditioned_input = torch.cat([speaker_embeddings, projected_embeddings], dim=1)
        
        # Generate audio tokens autoregressively
        audio_tokens = self.gpt.generate(
            cond_latents=conditioned_input,
            max_length=max_audio_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # ==================== Step 7: Decode to Waveform ====================
        
        # HiFiGAN vocoder
        audio = self.hifigan.inference(audio_tokens)
        
        # Convert to numpy
        audio = audio.squeeze().cpu().numpy()
        
        return audio, translated_ids
    
    def get_trainable_parameters(self, stage):
        """
        Get parameters for optimizer based on training stage
        
        Args:
            stage: 'projection' | 'translation' | 'end_to_end'
        
        Returns:
            List of parameter groups with learning rates
        """
        if stage == "projection":
            return [
                {'params': self.projection.parameters(), 'lr': 1e-4},
            ]
        
        elif stage == "translation":
            return [
                {'params': self.semantic_encoder.parameters(), 'lr': 1e-5},
                {'params': self.translation_decoder.parameters(), 'lr': 1e-5},
                {'params': self.projection.parameters(), 'lr': 5e-5},
            ]
        
        elif stage == "end_to_end":
            return [
                {'params': self.semantic_encoder.parameters(), 'lr': 1e-6},
                {'params': self.translation_decoder.parameters(), 'lr': 1e-6},
                {'params': self.projection.parameters(), 'lr': 1e-5},
                {'params': self.gpt.parameters(), 'lr': 2e-6},
            ]
        
        else:
            raise ValueError(f"Unknown stage: {stage}")
```

### Training Script
```python
# train_ti_xtts.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import argparse
from tqdm import tqdm

from ti_xtts.model import TranslationIntegratedXTTS


class TranslationTTSDataset(Dataset):
    """
    Dataset for training Translation-Integrated XTTS
    
    Expected data format:
    - CSV with columns: audio_path|source_text|speaker_id
    - Audio files: target language speech
    - source_text: English text
    """
    
    def __init__(
        self,
        metadata_file,
        nllb_tokenizer,
        dvae_model,
        max_text_length=300,
        max_audio_length=330750,
        target_lang="swh_Latn",
    ):
        self.metadata = self._load_metadata(metadata_file)
        self.nllb_tokenizer = nllb_tokenizer
        self.dvae = dvae_model
        self.max_text_length = max_text_length
        self.max_audio_length = max_audio_length
        self.target_lang = target_lang
        
        print(f"Loaded {len(self.metadata)} samples from {metadata_file}")
    
    def _load_metadata(self, metadata_file):
        """Load metadata CSV"""
        import pandas as pd
        
        samples = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    audio_path = parts[0]
                    text = parts[1]
                    speaker_id = parts[2]
                    
                    # Extract English text (remove language tags)
                    if '<eng>' in text:
                        english_text = text.split('<eng>')[1].split('<swa>')[0].strip()
                    else:
                        english_text = text
                    
                    samples.append({
                        'audio_path': audio_path,
                        'text': english_text,
                        'speaker_id': speaker_id,
                    })
        
        return samples
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata[idx]
        
        # Load and process audio
        import torchaudio
        audio, sr = torchaudio.load(sample['audio_path'])
        
        # Resample to 22.05kHz
        if sr != 22050:
            resampler = torchaudio.transforms.Resample(sr, 22050)
            audio = resampler(audio)
        
        # Convert to mel-spectrogram
        from TTS.tts.layers.xtts.audio_utils import wav_to_mel_clipping
        mel = wav_to_mel_clipping(
            audio.squeeze(0).numpy(),
            mel_norms_file=None,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
        )
        
        # Extract audio codes using DVAE
        with torch.no_grad():
            audio_codes = self.dvae.get_codes(torch.FloatTensor(mel).unsqueeze(0))
            audio_codes = audio_codes.squeeze(0)
        
        # Tokenize English text
        self.nllb_tokenizer.src_lang = "eng_Latn"
        text_encoding = self.nllb_tokenizer(
            sample['text'],
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'source_ids': text_encoding['input_ids'].squeeze(0),
            'source_mask': text_encoding['attention_mask'].squeeze(0),
            'speaker_mel': torch.FloatTensor(mel),
            'audio_codes': audio_codes,
            'text': sample['text'],
        }


def train_stage(
    model,
    train_loader,
    val_loader,
    stage,
    num_epochs,
    output_dir,
    device,
):
    """
    Train a specific stage
    
    Args:
        model: TranslationIntegratedXTTS model
        train_loader: Training data loader
        val_loader: Validation data loader
        stage: 'projection' | 'translation' | 'end_to_end'
        num_epochs: Number of epochs to train
        output_dir: Directory to save checkpoints
        device: torch device
    """
    print(f"\n{'='*60}")
    print(f"Training Stage: {stage.upper()}")
    print(f"{'='*60}\n")
    
    # Set training stage (freezes/unfreezes appropriate components)
    model.set_training_stage(stage)
    
    # Get optimizer parameters for this stage
    param_groups = model.get_trainable_parameters(stage)
    optimizer = AdamW(param_groups, weight_decay=0.01)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            source_ids = batch['source_ids'].to(device)
            source_mask = batch['source_mask'].to(device)
            speaker_mel = batch['speaker_mel'].to(device)
            audio_codes = batch['audio_codes'].to(device)
            
            # Forward pass with mixed precision
            with autocast():
                logits = model(
                    source_text_ids=source_ids,
                    source_attention_mask=source_mask,
                    speaker_wav=speaker_mel,
                    target_audio_codes=audio_codes,
                )
                
                # Compute loss
                # Shift targets: predict next token
                loss = criterion(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    audio_codes[:, 1:].reshape(-1),
                )
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                output_dir,
                f"best_model_{stage}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model to {checkpoint_path}")


def validate(model, val_loader, criterion, device):
    """Validation loop"""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            source_ids = batch['source_ids'].to(device)
            source_mask = batch['source_mask'].to(device)
            speaker_mel = batch['speaker_mel'].to(device)
            audio_codes = batch['audio_codes'].to(device)
            
            with autocast():
                logits = model(
                    source_text_ids=source_ids,
                    source_attention_mask=source_mask,
                    speaker_wav=speaker_mel,
                    target_audio_codes=audio_codes,
                )
                
                loss = criterion(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    audio_codes[:, 1:].reshape(-1),
                )
            
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_metadata', type=str, required=True)
    parser.add_argument('--val_metadata', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints/ti_xtts')
    parser.add_argument('--stage', type=str, default='projection',
                       choices=['projection', 'translation', 'end_to_end'])
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print("Initializing Translation-Integrated XTTS...")
    model = TranslationIntegratedXTTS(
        nllb_model_name="facebook/nllb-200-distilled-600M",
        xtts_config_path="checkpoints/base/config.json",
        xtts_checkpoint_path="checkpoints/base/model.pth",
        dvae_checkpoint_path="checkpoints/base/dvae.pth",
    )
    model = model.to(args.device)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = TranslationTTSDataset(
        args.train_metadata,
        model.nllb_tokenizer,
        model.dvae,
    )
    
    val_dataset = TranslationTTSDataset(
        args.val_metadata,
        model.nllb_tokenizer,
        model.dvae,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Train
    train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        stage=args.stage,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
```

### Inference Script
```python
# inference.py

import torch
import argparse
import soundfile as sf
from ti_xtts.model import TranslationIntegratedXTTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--text', type=str, required=True,
                       help='English text to translate and synthesize')
    parser.add_argument('--speaker_wav', type=str, required=True,
                       help='Reference audio for voice cloning')
    parser.add_argument('--target_lang', type=str, default='swh_Latn',
                       help='Target language code (e.g., swh_Latn for Swahili)')
    parser.add_argument('--output', type=str, default='output.wav',
                       help='Output audio file path')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = TranslationIntegratedXTTS()
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    # Generate
    print(f"Translating: {args.text}")
    print(f"Target language: {args.target_lang}")
    print(f"Speaker reference: {args.speaker_wav}")
    
    audio, translated_ids = model.inference(
        source_text=args.text,
        speaker_wav_path=args.speaker_wav,
        target_lang=args.target_lang,
    )
    
    # Decode translation to see what was generated
    translated_text = model.nllb_tokenizer.decode(
        translated_ids[0],
        skip_special_tokens=True
    )
    print(f"Translation: {translated_text}")
    
    # Save audio
    sf.write(args.output, audio, 22050)
    print(f"✓ Saved audio to {args.output}")


if __name__ == "__main__":
    main()
```

## Training Pipeline

### Stage 1: Projection Layer (2-3 days)

Train only the projection layer to learn format conversion.
```bash
python train_ti_xtts.py \
    --train_metadata data/train.csv \
    --val_metadata data/val.csv \
    --stage projection \
    --num_epochs 5 \
    --batch_size 16 \
    --output_dir checkpoints/ti_xtts/stage1
```

**What happens:**
- NLLB and XTTS are frozen
- Only projection layer learns
- Loss should decrease steadily
- Audio quality will be poor initially

**Expected loss:**
- Start: ~8.0
- End: ~4.0

### Stage 2: Translation Fine-tuning (1 week)

Fine-tune NLLB + projection to work better together.
```bash
# Load best checkpoint from stage 1
python train_ti_xtts.py \
    --train_metadata data/train.csv \
    --val_metadata data/val.csv \
    --stage translation \
    --num_epochs 10 \
    --batch_size 12 \
    --output_dir checkpoints/ti_xtts/stage2 \
    --resume_from checkpoints/ti_xtts/stage1/best_model_projection.pth
```

**What happens:**
- NLLB learns to produce embeddings optimized for TTS
- Projection layer continues improving
- XTTS remains frozen
- Audio quality improves

**Expected loss:**
- Start: ~4.0
- End: ~2.5

### Stage 3: End-to-End (2-3 weeks)

Fine-tune entire pipeline for best quality.
```bash
python train_ti_xtts.py \
    --train_metadata data/train.csv \
    --val_metadata data/val.csv \
    --stage end_to_end \
    --num_epochs 20 \
    --batch_size 8 \
    --output_dir checkpoints/ti_xtts/stage3 \
    --resume_from checkpoints/ti_xtts/stage2/best_model_translation.pth
```

**What happens:**
- Everything trains together (except HiFiGAN)
- Translation + TTS jointly optimized
- Best audio quality achieved

**Expected loss:**
- Start: ~2.5
- End: ~1.0-1.5

## Usage

### Training
```bash
# Stage 1
python train_ti_xtts.py \
    --train_metadata /path/to/train.csv \
    --val_metadata /path/to/val.csv \
    --stage projection \
    --num_epochs 5 \
    --batch_size 16

# Stage 2
python train_ti_xtts.py \
    --train_metadata /path/to/train.csv \
    --val_metadata /path/to/val.csv \
    --stage translation \
    --num_epochs 10 \
    --batch_size 12 \
    --resume_from checkpoints/ti_xtts/stage1/best_model_projection.pth

# Stage 3
python train_ti_xtts.py \
    --train_metadata /path/to/train.csv \
    --val_metadata /path/to/val.csv \
    --stage end_to_end \
    --num_epochs 20 \
    --batch_size 8 \
    --resume_from checkpoints/ti_xtts/stage2/best_model_translation.pth
```

### Inference
```bash
python inference.py \
    --checkpoint checkpoints/ti_xtts/stage3/best_model_end_to_end.pth \
    --text "Hello, how are you today?" \
    --speaker_wav reference_audio.wav \
    --target_lang swh_Latn \
    --output output.wav
```

### Python API
```python
from ti_xtts.model import TranslationIntegratedXTTS
import torch

# Load model
model = TranslationIntegratedXTTS()
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate
audio, translation = model.inference(
    source_text="The weather is beautiful today",
    speaker_wav_path="speaker.wav",
    target_lang="swh_Latn",  # Swahili
)

# Save
import soundfile as sf
sf.write('output.wav', audio, 22050)
```

## Supported Languages

Using NLLB-600M, the model supports 200+ languages. Main African languages in your dataset:

| Language | Code | NLLB Code |
|----------|------|-----------|
| Swahili | swa | `swh_Latn` |
| Xhosa | xho | `xho_Latn` |
| Igbo | ibo | `ibo_Latn` |
| Efik | efi | `efi_Latn` |

Full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

## Architecture Details

### Model Sizes

| Component | Parameters | Trainable (Stage 1) | Trainable (Stage 2) | Trainable (Stage 3) |
|-----------|------------|---------------------|---------------------|---------------------|
| NLLB Encoder | ~300M | ❌ | ✅ | ✅ |
| NLLB Decoder | ~300M | ❌ | ✅ | ✅ |
| Projection | ~10M | ✅ | ✅ | ✅ |
| XTTS GPT-2 | ~400M | ❌ | ❌ | ✅ |
| XTTS Perceiver | ~50M | ❌ | ❌ | ❌ |
| XTTS HiFiGAN | ~50M | ❌ | ❌ | ❌ |
| **Total** | ~1.1B | 10M | 610M | 1.01B |

### Memory Requirements

| Stage | Training | Inference |
|-------|----------|-----------|
| Stage 1 | ~24GB | ~8GB |
| Stage 2 | ~32GB | ~8GB |
| Stage 3 | ~40GB | ~8GB |

Recommendations:
- Training: A100 40GB or 80GB
- Inference: RTX 3090 (24GB) or better

### Training Time Estimates

With 2x A100 GPUs and 80k training samples:

| Stage | Epochs | Time per Epoch | Total Time |
|-------|--------|----------------|------------|
| Stage 1 | 5 | 2 hours | ~10 hours |
| Stage 2 | 10 | 4 hours | ~40 hours |
| Stage 3 | 20 | 6 hours | ~120 hours |

**Total training time: ~1 week**

## Differences from Pipeline Approach

### Pipeline (Option 1)
```python
# Separate models
translator = NLLBModel()
tts = XTTSModel()

# Two-step inference
swahili_text = translator.translate(english_text)
audio = tts.synthesize(swahili_text, speaker_wav)
```

**Pros:**
- Easy to implement
- Can swap components
- No training needed

**Cons:**
- Two separate models in memory
- No gradient flow between components
- Cant optimize translation for TTS quality
- Higher latency

### Integrated (TI-XTTS)
```python
# Single unified model
model = TranslationIntegratedXTTS()

# One-step inference
audio = model.inference(english_text, speaker_wav, target_lang)
```

**Pros:**
- Single model in memory
- End-to-end gradient flow
- Translation optimized for TTS
- Lower latency
- Better quality (potentially)

**Cons:**
- Complex implementation
- Requires training
- Harder to debug

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1:** Reduce batch size
```bash
--batch_size 4  # Instead of 8
```

**Solution 2:** Gradient accumulation
```bash
--batch_size 4 --grad_accum_steps 2  # Effective batch size = 8
```

**Solution 3:** Use mixed precision (already enabled)

### Issue: Loss not decreasing

**Stage 1:**
- Check learning rate (try 1e-3 to 1e-4)
- Increase batch size if possible
- Train for more epochs

**Stage 2:**
- Make sure stage 1 converged first
- Lower learning rate for NLLB (try 5e-6)
- Check gradient norms

**Stage 3:**
- Use very low learning rates (1e-6 for NLLB/GPT)
- Train for many epochs (20-50)
- Monitor validation loss carefully

### Issue: Poor audio quality

**Possible causes:**
1. **Projection layer not trained enough**
   - Train stage 1 for more epochs
   - Try larger projection network

2. **XTTS components degraded**
   - Keep HiFiGAN frozen always
   - Use lower learning rate for GPT-2 in stage 3

3. **Speaker conditioning issues**
   - Check speaker reference audio quality
   - Try different reference audios
   - Verify mel-spectrogram extraction

### Issue: Poor translation quality

**Possible causes:**
1. **NLLB not fine-tuned enough**
   - Train stage 2 longer
   - Check if forced_bos_token_id is correct

2. **Data quality issues**
   - Verify English text is clean
   - Check target audio matches text
   - Remove corrupted samples

## Monitoring Training

### Tensorboard
```bash
tensorboard --logdir checkpoints/ti_xtts
```

**Metrics to watch:**
- `train/loss`: Should decrease steadily
- `val/loss`: Should track train loss (gap = overfitting)
- `train/grad_norm`: Should be stable (< 10.0)

### WandB (Optional)
```python
import wandb

wandb.init(project="ti-xtts", name=f"stage_{args.stage}")
wandb.watch(model)
```

## Future Improvements

### 1. Better Projection Layer
Current: Simple 2-layer MLP

Potential improvements:
- Transformer-based projection
- Attention mechanism between NLLB and GPT-2
- Learnable bottleneck

### 2. Multi-task Learning
Train on multiple tasks simultaneously:
- Translation (English → Target text)
- TTS (Target text → Audio)
- Voice cloning consistency

### 3. Streaming Inference
Current: Batch processing

Goal: Real-time streaming
- Chunk-based translation
- Incremental audio generation
- < 200ms latency

### 4. More Languages
- Extend to all 200+ NLLB languages
- Language-specific fine-tuning
- Zero-shot transfer to unseen languages

## Citation

If you use this work, please cite:
```bibtex
@misc{ti-xtts-2024,
  title={Translation-Integrated XTTS: End-to-End Multilingual Speech Translation with Voice Cloning},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/ti-xtts}}
}
```

## Related Work

- **XTTS**: https://github.com/coqui-ai/TTS
- **NLLB**: https://github.com/facebookresearch/fairseq/tree/nllb
- **SeamlessM4T**: https://github.com/facebookresearch/seamless_communication
- **VALL-E X**: https://arxiv.org/abs/2303.03926

## License

This project combines:
- XTTS (Mozilla Public License 2.0)
- NLLB (CC-BY-NC 4.0)

Please check individual licenses for commercial use.

## Acknowledgments

- Coqui AI for XTTS
- Meta AI for NLLB
- The open-source community

---

**Questions or issues?** Open an issue on GitHub or contact [your-email]