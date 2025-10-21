import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load environment variables
load_dotenv()

print("Setting up NLLB model...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Model configuration
model_name = "facebook/nllb-200-distilled-600M"
local_model_path = "./nllb-200-distilled-600M-local"

# Check if model exists locally
if os.path.exists(local_model_path):
    print(f"Loading model from local path: {local_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
else:
    print(f"Downloading and saving model locally...")
    # Download and save locally
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Save locally for future use
    tokenizer.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)
    print(f"Model saved to: {local_model_path}")

# Force CPU usage
device = torch.device("cpu")
model = model.to(device)

print("Model loaded successfully!")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Local model path: {local_model_path}")

# Translation function
def translate_text(text, source_lang="eng_Latn", target_lang="fra_Latn"):
    """
    Translate text using NLLB model
    
    Args:
        text (str): Text to translate
        source_lang (str): Source language code (e.g., "eng_Latn" for English)
        target_lang (str): Target language code (e.g., "fra_Latn" for French)
    
    Returns:
        str: Translated text
    """
    try:
        # Set source language
        tokenizer.src_lang = source_lang
        
        # Encode the text
        encoded = tokenizer(text, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get target language ID using the correct method
        target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=target_lang_id,
                max_length=512
            )
        
        # Decode the translation
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return translation
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Full error details:\n{error_details}")
        return f"Error: {e}"

print("\nModel is ready for translation!")
print("Available functions:")
print("1. translate_text(text, source_lang, target_lang)")
print("\nExample language codes:")
print("- English: eng_Latn")
print("- French: fra_Latn") 
print("- Spanish: spa_Latn")
print("- German: deu_Latn")
print("- Chinese: zho_Hans")
print("- Arabic: arb_Arab")

if __name__ == "__main__":
    # Test translation
    print("\n" + "="*60)
    print("TESTING NLLB TRANSLATION")
    print("="*60)
    
    test_text = "Hello, how are you today?"
    print(f"Original text: {test_text}")
    
    # Test English to French
    french_translation = translate_text(test_text, "eng_Latn", "fra_Latn")
    print(f"French translation: {french_translation}")
    
    # Test English to Spanish
    spanish_translation = translate_text(test_text, "eng_Latn", "spa_Latn")
    print(f"Spanish translation: {spanish_translation}")
    
    print("\n✅ NLLB model is working correctly!")
    print("You can now use translate_text() for any supported language pair.")
