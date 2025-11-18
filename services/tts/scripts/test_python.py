import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Device configuration
device = "cpu"  # Force CPU to avoid CUDA device busy error

# Model paths
xtts_checkpoint = "checkpoints/XTTS_v2.0_original_model_files/model.pth"
xtts_config = "checkpoints/XTTS_v2.0_original_model_files/config.json"
xtts_vocab = "checkpoints/XTTS_v2.0_original_model_files/vocab.json"

# Load model
config = XttsConfig()
config.load_json(xtts_config)
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
XTTS_MODEL.to(device)

print("Model loaded successfully!")

# Create a dummy reference audio file (3 seconds of very quiet white noise at 22050 Hz)
import numpy as np
np.random.seed(42)  # For reproducibility
dummy_audio = np.random.normal(0, 0.01, 22050 * 3).astype(np.float32)  # 3 seconds of quiet noise
torchaudio.save("dummy_ref.wav", torch.tensor(dummy_audio).unsqueeze(0), 22050)
print("Created dummy reference audio file with quiet noise")

# Inference
tts_text = "Hello, this is a test of the text to speech system."
speaker_audio_file = "dummy_ref.wav"
lang = "en"  # Use English instead of Vietnamese for better compatibility

gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)

tts_texts = sent_tokenize(tts_text)

wav_chunks = []
for text in tqdm(tts_texts):
    wav_chunk = XTTS_MODEL.inference(
        text=text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.1,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=10,
        top_p=0.3,
    )
    wav_chunks.append(torch.tensor(wav_chunk["wav"]))

out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()

# Save the output audio file
torchaudio.save("output_tts.wav", out_wav, 24000)
print(f"Generated TTS audio saved as 'output_tts.wav'")
print(f"Audio shape: {out_wav.shape}")
print("TTS generation completed successfully!")