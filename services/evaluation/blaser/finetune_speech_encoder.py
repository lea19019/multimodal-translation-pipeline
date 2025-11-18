"""
Fine-tune SONAR speech encoders - COMPLETE STANDALONE VERSION
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from tqdm import tqdm
import os
import pandas as pd
import torchaudio

from fairseq2.models import load_model
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

# Config
CHECKPOINT_DIR = Path("./checkpoints")
DATA_ROOT = Path("/home/vacl2/multimodal_translation/services/data/languages")

LANG_CONFIG = {
    # 'swh': {'name': 'swahili', 'base_encoder': 'swh', 'text_code': 'swh_Latn'}, # DONE
    # 'xho': {'name': 'xhosa', 'base_encoder': 'swh', 'text_code': 'xho_Latn'},
    # 'ibo': {'name': 'igbo', 'base_encoder': 'eng', 'text_code': 'ibo_Latn'},
    'efi': {'name': 'efik', 'base_encoder': 'eng', 'text_code': 'ibo_Latn'}  # Use Igbo as proxy for Efik
}


class SpeechTextDataset(Dataset):
    def __init__(self, csv_path, target_sr=16000):
        self.df = pd.read_csv(csv_path, sep='|')
        self.target_sr = target_sr
        
        valid_indices = []
        for idx, row in self.df.iterrows():
            if Path(row['audio_file']).exists():
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, sr = torchaudio.load(row['audio_file'])
        
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        return {
            'audio': audio.squeeze(0),
            'text': row['text']
        }


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x['audio'].shape[0], reverse=True)
    max_len = batch[0]['audio'].shape[0]
    
    audios = []
    texts = []
    lengths = []
    
    for item in batch:
        audio = item['audio']
        length = audio.shape[0]
        
        if length < max_len:
            padding = torch.zeros(max_len - length)
            audio = torch.cat([audio, padding])
        
        audios.append(audio)
        texts.append(item['text'])
        lengths.append(length)
    
    return {
        'audio': torch.stack(audios),
        'text': texts,
        'lengths': torch.tensor(lengths)
    }


def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    return rank, world_size, local_rank


class SpeechEncoderTrainer:
    
    def __init__(self, lang_code, learning_rate=1e-5, rank=0, world_size=1, gradient_accumulation_steps=1):
        self.lang_code = lang_code
        self.config = LANG_CONFIG[lang_code]
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        if self.is_main:
            print(f"\nTraining {self.config['name']}")
            print(f"  Device: {self.device}")
        
        # Load speech encoder using pipeline (handles preprocessing)
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
        
        encoder_name = f"sonar_speech_encoder_{self.config['base_encoder']}"
        
        # Load the pipeline to get the model
        self.speech_pipeline = SpeechToEmbeddingModelPipeline(
            encoder=encoder_name,
            device=self.device
        )
        
        # Extract the actual model for training
        # The model has built-in waveform->fbank conversion
        self.speech_model = self.speech_pipeline.model
        self.speech_model.train()
        
        # Wrap with DDP if using multiple GPUs
        if world_size > 1:
            self.speech_model = DDP(self.speech_model, device_ids=[rank])
        
        # Load text encoder (teacher)
        self.text_encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device
        )
        for param in self.text_encoder.model.parameters():
            param.requires_grad = False
        
        # Optimizer
        params = [p for p in self.speech_model.parameters() if p.requires_grad]
        print(f"  Trainable parameters: {sum(p.numel() for p in params):,}")
        
        self.optimizer = AdamW(params, lr=learning_rate, weight_decay=0.01)
        self.criterion = nn.MSELoss()
        self.best_loss = float('inf')
    
    def get_speech_embedding(self, audio_batch, lengths=None):
        """
        Get speech embeddings with gradient support.
        Manually compute fbank features using torchaudio.
        audio_batch: (batch_size, time) tensor of raw waveforms @ 16kHz
        """
        import torchaudio
        from fairseq2.data import Collater
        from sonar.inference_pipelines.utils import extract_sequence_batch
        
        # Step 1: Convert to fbank features using torchaudio directly
        fbank_list = []
        for i in range(audio_batch.shape[0]):
            # Get audio for this sample
            if lengths is not None:
                audio = audio_batch[i, :lengths[i]]
            else:
                audio = audio_batch[i]
            
            # Compute fbank features using torchaudio
            # audio is (time,), we need (channels, time) = (1, time)
            audio_2d = audio.unsqueeze(0).to(self.device)
            
            # Compute 80-dim fbank features
            fbank = torchaudio.compliance.kaldi.fbank(
                audio_2d,
                num_mel_bins=80,
                sample_frequency=16000,
                frame_length=25,  # 25ms window
                frame_shift=10,   # 10ms hop
            )
            # fbank shape: (time_steps, 80)
            
            # Create dict in the format expected by collater
            fbank_dict = {
                "fbank": fbank,  # (time_steps, 80)
                "sample_rate": 16000.0,
                "format": -1,
            }
            fbank_list.append(fbank_dict)
        
        # Step 2: Collate into batch (pad to same length)
        collater = Collater(pad_value=0, pad_to_multiple=2)
        batch_dict = collater(fbank_list)
        
        # Step 3: Extract sequences from the "fbank" key
        fbank_seqs, seqs_layout = extract_sequence_batch(batch_dict["fbank"], self.device)
        
        # Step 4: Forward through model
        output = self.speech_model(fbank_seqs, seqs_layout)
        
        embeddings = output.sentence_embeddings
        
        # Clean up intermediate tensors
        del fbank_list, batch_dict, fbank_seqs, seqs_layout, output
        
        return embeddings
        
        # Step 4: Forward through model (NO @torch.inference_mode() here!)
        output = self.speech_model(fbank_seqs, seqs_layout)
        
        return output.sentence_embeddings
    
    def get_text_embedding(self, texts):
        """Get text embeddings from teacher model (no gradients needed)."""
        with torch.no_grad():
            emb = self.text_encoder.predict(texts, source_lang=self.config['text_code'])
        # Detach and clone to ensure it's a regular tensor, not an inference tensor
        return emb.detach().clone().to(self.device)
    
    def train_step(self, batch, accumulation_step):
        audio = batch['audio'].to(self.device)
        texts = batch['text']
        lengths = batch['lengths']
        
        teacher_emb = self.get_text_embedding(texts)
        student_emb = self.get_speech_embedding(audio, lengths)
        
        loss = self.criterion(student_emb, teacher_emb)
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        loss.backward()
        
        # Only update weights every N steps
        if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        loss_val = loss.item() * self.gradient_accumulation_steps  # Unscale for logging
        
        # Aggressive memory cleanup
        del audio, texts, lengths, teacher_emb, student_emb, loss
        torch.cuda.empty_cache()
        
        return loss_val
        del teacher_emb, student_emb, audio
        torch.cuda.empty_cache()
        
        return loss.item()
    
    @torch.no_grad()
    def eval_step(self, batch):
        audio = batch['audio'].to(self.device)
        texts = batch['text']
        lengths = batch['lengths']
        
        teacher_emb = self.get_text_embedding(texts)
        student_emb = self.get_speech_embedding(audio, lengths)
        loss = self.criterion(student_emb, teacher_emb)
        
        return loss.item()
    
    def train_epoch(self, loader):
        model = self.speech_model.module if self.world_size > 1 else self.speech_model
        model.train()
        total_loss = 0
        
        self.optimizer.zero_grad()  # Initialize at start of epoch
        
        pbar = tqdm(loader, disable=not self.is_main)
        for step, batch in enumerate(pbar):
            loss = self.train_step(batch, step)
            total_loss += loss
            if self.is_main:
                pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        return total_loss / len(loader)
    
    @torch.no_grad()
    def evaluate(self, loader):
        model = self.speech_model.module if self.world_size > 1 else self.speech_model
        model.eval()
        total_loss = 0
        
        for batch in tqdm(loader, disable=not self.is_main):
            loss = self.eval_step(batch)
            total_loss += loss
        
        return total_loss / len(loader)
    
    def save(self, epoch, train_loss, eval_loss):
        if not self.is_main:
            return
        
        save_dir = CHECKPOINT_DIR / f"finetuned_{self.lang_code}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model = self.speech_model.module if self.world_size > 1 else self.speech_model
        
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss
        }, save_dir / f"model_epoch{epoch}.pt")
        
        print(f"  Saved: finetuned_{self.lang_code}/model_epoch{epoch}.pt")
    
    def train(self, train_loader, eval_loader, num_epochs):
        for epoch in range(1, num_epochs + 1):
            if self.is_main:
                print(f"\nEpoch {epoch}/{num_epochs}")
            
            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)
            
            train_loss = self.train_epoch(train_loader)
            eval_loss = self.evaluate(eval_loader)
            
            if self.is_main:
                print(f"Train: {train_loss:.4f} | Eval: {eval_loss:.4f}")
            
            self.save(epoch, train_loss, eval_loss)
            
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                if self.is_main:
                    print(f"  New best!")
                self.save('best', train_loss, eval_loss)


def create_dataloaders(lang_code, batch_size, num_workers, world_size, rank):
    config = LANG_CONFIG[lang_code]
    lang_dir = DATA_ROOT / config['name']
    
    train_dataset = SpeechTextDataset(lang_dir / "metadata_train.csv")
    eval_dataset = SpeechTextDataset(lang_dir / "metadata_eval.csv")
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, eval_loader


def main():
    rank, world_size, _ = setup_distributed()
    
    if rank == 0:
        print("Fine-tuning SONAR Speech Encoders")
        print(f"GPUs: {world_size}")
    
    config = {
        'num_epochs': 1,  # Reduced to 1 for faster training
        'batch_size': 2,  # Reduced to 2 to avoid OOM
        'learning_rate': 1e-5,
        'num_workers': 8,
        'gradient_accumulation_steps': 4  # Effective batch_size = 8
    }
    
    for lang_code in ['efi']:#['swh', 'xho', 'ibo', 'efi']:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Training: {LANG_CONFIG[lang_code]['name'].upper()}")
        
        try:
            train_loader, eval_loader = create_dataloaders(
                lang_code,
                config['batch_size'],
                config['num_workers'],
                world_size,
                rank
            )
            
            trainer = SpeechEncoderTrainer(
                lang_code, 
                config['learning_rate'], 
                rank, 
                world_size,
                config.get('gradient_accumulation_steps', 1)
            )
            trainer.train(train_loader, eval_loader, config['num_epochs'])
            
        except Exception as e:
            if rank == 0:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()