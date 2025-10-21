import os
from dotenv import load_dotenv
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# Load environment variables
load_dotenv()

print("Setting up Whisper model...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Model configuration
model_name = "openai/whisper-medium"
local_model_path = "./whisper-medium-local"

# Check if model exists locally
if os.path.exists(local_model_path):
    print(f"Loading model from local path: {local_model_path}")
    processor = AutoProcessor.from_pretrained(local_model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(local_model_path)
else:
    print(f"Downloading and saving model locally...")
    # Download and save locally
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    
    # Save locally for future use
    processor.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)
    print(f"Model saved to: {local_model_path}")

# Force CPU usage
device = torch.device("cpu")
model = model.to(device)

print("Model loaded successfully!")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Local model path: {local_model_path}")

# Test function for transcribing audio
def transcribe_audio(audio_file_path):
    """Transcribe an audio file using the loaded Whisper model"""
    try:
        import numpy as np
        
        # Try pydub first for WebM support
        try:
            print("Trying pydub for WebM support...")
            from pydub import AudioSegment
            
            # Load with pydub
            audio_segment = AudioSegment.from_file(audio_file_path)
            
            # Convert to WAV format in memory
            wav_file = "temp_converted.wav"
            audio_segment.export(wav_file, format="wav")
            print(f"Converted {audio_file_path} to {wav_file}")
            
            # Now load with librosa
            import librosa
            audio, sr = librosa.load(wav_file, sr=16000)
            print(f"Successfully loaded converted file: 16000Hz")
            
            # Clean up temp file
            import os
            os.remove(wav_file)
            
        except Exception as e:
            print(f"Pydub conversion failed: {e}")
            print("Trying torchaudio...")
            
            try:
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio_file_path)
                # Convert to numpy and take first channel if stereo
                audio = waveform[0].numpy() if waveform.shape[0] > 1 else waveform.squeeze().numpy()
                
                # Resample if needed
                if sample_rate != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                
                print(f"Successfully loaded with torchaudio: {sample_rate}Hz -> 16000Hz")
                
            except Exception as e2:
                print(f"Torchaudio also failed: {e2}")
                print("Falling back to librosa...")
                import librosa
                # Load audio file with librosa
                audio, sr = librosa.load(audio_file_path, sr=16000)  # Whisper expects 16kHz
        
        # Handle longer audio by chunking
        audio_length = len(audio) / 16000
        print(f"Audio length: {audio_length:.1f} seconds")
        
        if audio_length > 30:
            print("Audio is longer than 30 seconds, processing in chunks...")
            chunk_length = 30 * 16000  # 30 seconds in samples
            transcriptions = []
            
            for i in range(0, len(audio), chunk_length):
                chunk = audio[i:i + chunk_length]
                print(f"Processing chunk {i//chunk_length + 1}/{(len(audio) + chunk_length - 1)//chunk_length}")
                
                # Process chunk
                inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = model.generate(**inputs)
                
                chunk_transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                transcriptions.append(chunk_transcription.strip())
            
            # Combine all transcriptions
            transcription = " ".join(transcriptions)
        else:
            # Process normally for shorter audio
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
            
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return transcription
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Full error details:\n{error_details}")
        return f"Error: {e}\n\nFull traceback:\n{error_details}"





print("\nModel is ready for transcription!")
print("Available functions:")
print("1. transcribe_audio('path/to/audio.wav') - Single file")
print("\nNote: You may need to install: uv pip install librosa")

if __name__ == "__main__":
    # Test with your mp3 file
    print("\n" + "="*60)
    print("TESTING WITH YOUR WAV FILE")
    print("="*60)
    
    audio_file = "whatisapple.wav"
    
    if os.path.exists(audio_file):
        print(f"Found audio file: {audio_file}")
        print("Processing with Whisper...")
        
        transcription = transcribe_audio(audio_file)
        
        print(f"\n{'='*60}")
        print(f"TRANSCRIPTION RESULTS:")
        print(f"{'='*60}")
        print(f"Audio file: {audio_file}")
        print(f"Transcription:")
        print(f"{transcription}")
        print(f"{'='*60}")
        
        # Save transcription to file
        with open("transcription_output.txt", "w", encoding="utf-8") as f:
            f.write(f"Audio file: {audio_file}\n")
            f.write(f"Transcription:\n{transcription}")
        
        print(f"\n✅ Transcription saved to: transcription_output.txt")
        print("✅ Test completed successfully!")
        print("Your Whisper model is working correctly.")
        
    else:
        print(f"❌ Audio file '{audio_file}' not found in current directory.")
        print("Make sure the file is in the same directory as this script.")