import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# Load environment variables
load_dotenv()

print("Setting up NLLB model...")
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
    # Download and save locally
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Save locally for future use
    tokenizer.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)
    print(f"Model saved to: {local_model_path}")

# Add Efik token
tokenizer.add_tokens(["efi_Latn"])
model.resize_token_embeddings(len(tokenizer))

# Load your data files
with open("/home/vacl2/groups/grp_mtlab/projects/project-multimodal-pipeline/src.txt", "r", encoding="utf-8") as f:
    english = [line.strip() for line in f]

with open("/home/vacl2/groups/grp_mtlab/projects/project-multimodal-pipeline/tgt.txt", "r", encoding="utf-8") as f:
    efik = [line.strip() for line in f]

# Create full dataset
full_dataset = Dataset.from_dict({"english": english, "efik": efik})

# Split into train (90%) and eval (10%)
split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# Tokenize
def preprocess(examples):
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "efi_Latn"
    inputs = tokenizer(examples['english'], truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['efik'], truncation=True, max_length=128)
    inputs['labels'] = labels['input_ids']
    return inputs

train_dataset = train_dataset.map(preprocess, batched=True)
eval_dataset = eval_dataset.map(preprocess, batched=True)

# Training arguments
args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints/efik_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_steps=500,
    eval_steps=500,                   # Evaluate every 500 steps
    logging_steps=50,
    logging_first_step=True,
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,     # Load best checkpoint at end
    metric_for_best_model="eval_loss", # Use eval loss to pick best
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
)

# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,         # Add eval dataset
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)


trainer.train()

# Save
model.save_pretrained("./checkpoints/efik_finetuned_v2")
tokenizer.save_pretrained("./checkpoints/efik_finetuned_v2")