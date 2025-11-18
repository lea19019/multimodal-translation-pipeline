"""
Prepare data for fine-tuning SONAR speech encoders
"""

import pandas as pd
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

DATA_ROOT = Path("/home/vacl2/multimodal_translation/services/data/languages")

LANG_CONFIG = {
    'swh': {'name': 'swahili', 'base_encoder': 'swh', 'text_code': 'swh_Latn'},
    'efi': {'name': 'efik', 'base_encoder': 'eng', 'text_code': None},
    'xho': {'name': 'xhosa', 'base_encoder': 'swh', 'text_code': 'xho_Latn'},
    'ibo': {'name': 'igbo', 'base_encoder': 'eng', 'text_code': 'ibo_Latn'}
}


class SpeechTextDataset(Dataset):
    """Dataset for speech-text pairs"""
    
    def __init__(self, csv_path, target_sr=16000):
        self.df = pd.read_csv(csv_path, sep='|')
        self.target_sr = target_sr
        
        # Verify files exist
        valid_indices = []
        for idx, row in self.df.iterrows():
            if Path(row['audio_file']).exists():
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"Loaded {len(self.df)} valid samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio
        audio, sr = torchaudio.load(row['audio_file'])
        
        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        return {
            'audio': audio.squeeze(0),  # [T]
            'text': row['text'],
            'audio_path': row['audio_file']
        }


def load_language_data(lang_code, split='train'):
    """Load data for a specific language and split"""
    
    config = LANG_CONFIG[lang_code]
    lang_dir = DATA_ROOT / config['name']
    csv_path = lang_dir / f"metadata_{split}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata not found: {csv_path}")
    
    dataset = SpeechTextDataset(csv_path)
    return dataset


def collate_fn(batch):
    """Collate function for DataLoader - handles variable length audio"""
    
    # Sort by audio length (descending)
    batch = sorted(batch, key=lambda x: x['audio'].shape[0], reverse=True)
    
    # Get max length
    max_len = batch[0]['audio'].shape[0]
    
    # Pad all audio to max length
    audios = []
    texts = []
    lengths = []
    
    for item in batch:
        audio = item['audio']
        length = audio.shape[0]
        
        # Pad
        if length < max_len:
            padding = torch.zeros(max_len - length)
            audio = torch.cat([audio, padding])
        
        audios.append(audio)
        texts.append(item['text'])
        lengths.append(length)
    
    return {
        'audio': torch.stack(audios),  # [B, T]
        'text': texts,
        'lengths': torch.tensor(lengths)
    }


def create_dataloaders(lang_code, batch_size=8, num_workers=4):
    """Create train/eval dataloaders for a language"""
    
    train_dataset = load_language_data(lang_code, 'train')
    eval_dataset = load_language_data(lang_code, 'eval')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, eval_loader


def get_data_stats(lang_code):
    """Get statistics about the dataset"""
    
    config = LANG_CONFIG[lang_code]
    lang_dir = DATA_ROOT / config['name']
    
    stats = {}
    for split in ['train', 'eval', 'test']:
        csv_path = lang_dir / f"metadata_{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep='|')
            
            # Check how many files exist
            existing = sum(Path(f).exists() for f in df['audio_file'])
            
            stats[split] = {
                'total': len(df),
                'existing': existing,
                'missing': len(df) - existing
            }
    
    return stats


if __name__ == "__main__":
    print("Data Preparation for SONAR Fine-tuning")
    print("=" * 60)
    
    # Show stats for each language
    for lang_code, config in LANG_CONFIG.items():
        print(f"\n{config['name'].upper()} ({lang_code}):")
        print(f"  Base encoder: {config['base_encoder']}")
        print(f"  Text code: {config['text_code']}")
        
        try:
            stats = get_data_stats(lang_code)
            for split, counts in stats.items():
                print(f"  {split}: {counts['existing']}/{counts['total']} files")
                if counts['missing'] > 0:
                    print(f"    ⚠ {counts['missing']} files missing")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("\nTest loading a batch:")
    
    # Test with Swahili
    try:
        train_loader, eval_loader = create_dataloaders('swh', batch_size=2)
        batch = next(iter(train_loader))
        
        print(f"  Audio batch shape: {batch['audio'].shape}")
        print(f"  Number of texts: {len(batch['text'])}")
        print(f"  Sample text: {batch['text'][0][:50]}...")
        print(f"  Audio lengths: {batch['lengths']}")
        print("\n✓ Data loading works!")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")