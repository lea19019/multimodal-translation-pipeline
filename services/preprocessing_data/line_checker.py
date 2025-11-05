import os
import csv
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.vits_config import VitsConfig

# --- Configuration ---

# 1. IMPORTANT: Set this to the absolute path of your metadata file
metadata_file_path = "/grphome/grp_mtlab/projects/project-speech/african_tts/data/metadata.csv"

# 2. IMPORTANT: Set this to the absolute path of the config.json for the model you are finetuning.
#    When you run training, Coqui TTS downloads the model. You can find the path in your home
#    directory, something like: /home/your_user/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/config.json
#    Make sure this path is correct.
model_config_path = "/grphome/grp_mtlab/projects/project-speech/african_tts/TTS/recipes/multilingual/cml_yourtts/YourTTS-CML-TTS-July-07-2025_04+53PM-dbf1a08a/config.json"

problematic_lines = []

print("--- Advanced Metadata and Tokenizer Validation ---")

# --- Step 1: Basic File and Row Validation ---
print(f"\n[1/3] Checking for basic errors in: {metadata_file_path}")
try:
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        # Print length of file
        
        reader = csv.reader(f, delimiter='|')
        print(f"Length of file: {len(list(reader))}")
        for line_number, row in enumerate(reader, 1):
            if len(row) != 5:
                problematic_lines.append(f"Line {line_number}: Malformed row. Expected 5 columns, but found {len(row)}. Content: '{'|'.join(row)}'")
                continue
            transcription = row[1]
            if not transcription.strip():
                audio_file = row[0]
                problematic_lines.append(f"Line {line_number}: Empty transcription for audio file '{audio_file}'.")
except FileNotFoundError:
    print(f"\n[!!!] ERROR: The metadata file was not found at: {metadata_file_path}")
    exit()
except Exception as e:
    print(f"\n[!!!] An unexpected error occurred while reading the CSV: {e}")
    exit()

if problematic_lines:
    print("\n[!] Found basic formatting errors:")
    for problem in problematic_lines:
        print(f"  - {problem}")
    print("\nPlease fix these basic errors before proceeding.")
    exit()
else:
    print("[+] No basic formatting errors found.")


# --- Step 2: Tokenizer Validation ---
print(f"\n[2/3] Loading tokenizer from: {model_config_path}")
try:
    # Load the model config to initialize the tokenizer correctly
    config = VitsConfig()
    config.load_json(model_config_path)
    
    # Initialize the tokenizer from the model's config
    tokenizer, config = TTSTokenizer.init_from_config(config)
    print("[+] Tokenizer loaded successfully.")

except FileNotFoundError:
    print(f"\n[!!!] ERROR: The model config file was not found at: {model_config_path}")
    print("Please make sure the path is correct and you have run training at least once to download the model.")
    exit()
except Exception as e:
    print(f"\n[!!!] An unexpected error occurred while loading the tokenizer: {e}")
    exit()


print("\n[3/3] Checking each line for tokenizer errors...")
try:
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for line_number, row in enumerate(reader, 1):
            transcription = row[1]
            # Use the tokenizer to convert text to a sequence of IDs
            token_ids = tokenizer.text_to_ids(transcription)
            print(line_number)
            
            # Check if the tokenizer produced an empty output
            if not token_ids:
                audio_file = row[0]
                problematic_lines.append(f"Line {line_number}: Transcription '{transcription}' resulted in an EMPTY token sequence for audio file '{audio_file}'. This line is likely the cause of the error.")

except Exception as e:
    print(f"\n[!!!] An unexpected error occurred during tokenization check: {e}")
    exit()


# --- Final Report ---
if problematic_lines:
    print("\n[!] Found problematic lines that cause tokenizer errors:")
    for problem in problematic_lines:
        print(f"  - {problem}")
    print("\nThis likely means the text contains only characters that are not in the model's vocabulary.")
    print("Please remove these lines from your metadata.csv file and try training again.")
else:
    print("\n[+] Success! All lines in the metadata file were tokenized correctly.")
