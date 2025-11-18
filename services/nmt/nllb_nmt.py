import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset, concatenate_datasets

# Load environment variables
load_dotenv()

print("Setting up NLLB model for multi-language fine-tuning...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Model configuration
model_name = "facebook/nllb-200-distilled-600M"
local_model_path = "./checkpoints/base"

# Check if model exists locally
if os.path.exists(local_model_path):
    print(f"Loading model from local path: {local_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
else:
    print(f"Downloading and saving model locally...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    tokenizer.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)
    print(f"Model saved to: {local_model_path}")

# Add Efik language token (efi_Latn is NOT in base NLLB-200)
print("Adding efi_Latn as a special token...")
tokenizer.add_special_tokens({'additional_special_tokens': ['efi_Latn']})
model.resize_token_embeddings(len(tokenizer))
print(f"New vocab size: {len(tokenizer)}")
print(f"efi_Latn token ID: {tokenizer.convert_tokens_to_ids('efi_Latn')}")

# Language configurations
# Format: (language_code, source_file, target_file)
language_configs = [
    ("efi_Latn", "/home/vacl2/groups/grp_mtlab/projects/project-multimodal-pipeline/efik/src.txt", 
                 "/home/vacl2/groups/grp_mtlab/projects/project-multimodal-pipeline/efik/tgt.txt"),
    ("swh_Latn", "/home/vacl2/groups/grp_mtlab/projects/project-multimodal-pipeline/swahili/src.txt", 
                 "/home/vacl2/groups/grp_mtlab/projects/project-multimodal-pipeline/swahili/tgt.txt"),
    ("ibo_Latn", "/home/vacl2/groups/grp_mtlab/projects/project-multimodal-pipeline/igbo/src.txt", 
                 "/home/vacl2/groups/grp_mtlab/projects/project-multimodal-pipeline/igbo/tgt.txt"),
    ("xho_Latn", "/home/vacl2/groups/grp_mtlab/projects/project-multimodal-pipeline/xhosa/src.txt", 
                 "/home/vacl2/groups/grp_mtlab/projects/project-multimodal-pipeline/xhosa/tgt.txt"),
]

# Verify that Swahili, Igbo, and Xhosa are in the tokenizer
print("\nVerifying language codes in tokenizer:")
for lang_code, _, _ in language_configs:
    if lang_code != "efi_Latn":  # Skip Efik since we just added it
        if lang_code in tokenizer.additional_special_tokens or tokenizer.convert_tokens_to_ids(lang_code) != tokenizer.unk_token_id:
            print(f"✓ {lang_code} found in tokenizer")
        else:
            print(f"✗ WARNING: {lang_code} not found in tokenizer!")

# Load and combine all language datasets
all_datasets = []

for lang_code, src_file, tgt_file in language_configs:
    print(f"\nLoading {lang_code} data...")
    
    with open(src_file, "r", encoding="utf-8") as f:
        english = [line.strip() for line in f]
    
    with open(tgt_file, "r", encoding="utf-8") as f:
        target_lang = [line.strip() for line in f]
    
    # Create dataset with language code included
    dataset = Dataset.from_dict({
        "english": english,
        "target": target_lang,
        "target_lang": [lang_code] * len(english)
    })
    
    all_datasets.append(dataset)
    print(f"  Loaded {len(dataset)} sentence pairs for {lang_code}")

# Combine all datasets
print("\nCombining datasets...")
combined_dataset = concatenate_datasets(all_datasets)
print(f"Total combined samples: {len(combined_dataset)}")

# Split into train (90%) and eval (10%)
split_dataset = combined_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"\nTraining samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# Tokenize function that handles multiple target languages
def preprocess(examples):
    tokenizer.src_lang = "eng_Latn"
    
    # Tokenize source (English)
    inputs = tokenizer(examples['english'], truncation=True, max_length=128)
    
    # Tokenize targets with their respective language codes
    labels_list = []
    for target_text, target_lang in zip(examples['target'], examples['target_lang']):
        tokenizer.tgt_lang = target_lang
        with tokenizer.as_target_tokenizer():
            label = tokenizer(target_text, truncation=True, max_length=128)
        labels_list.append(label['input_ids'])
    
    inputs['labels'] = labels_list
    return inputs

print("\nTokenizing datasets...")
train_dataset = train_dataset.map(
    preprocess, 
    batched=True, 
    batch_size=1000,  # Increased batch size for faster processing
    # num_proc=4,  # Use multiple processes for tokenization
    # remove_columns=train_dataset.column_names,  # Remove original columns
    # desc="Tokenizing training data"
)

eval_dataset = eval_dataset.map(
    preprocess, 
    batched=True, 
    batch_size=1000,
    # num_proc=4,
    # remove_columns=eval_dataset.column_names,
    # desc="Tokenizing evaluation data"
)

# Training arguments
args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints/multilang_finetuned_running",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_steps=5000,
    eval_steps=5000,
    logging_steps=50,
    logging_first_step=True,
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    gradient_accumulation_steps=2,  # Optional: if memory is tight
)

# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

print("\nStarting training...")
trainer.train()

# Save final model
final_save_path = "./checkpoints/multilang_finetuned_final"
print(f"\nSaving final model to {final_save_path}...")
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

print("\n✓ Training completed successfully!")
print(f"Model saved to: {final_save_path}")