import argparse
import yaml
import os
import json
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

def parse_args():
    parser = argparse.ArgumentParser(description="Training Pipeline for Llama-3.1")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("1. Initializing Model and Tokenizer via Unsloth...")
    max_seq_length = config['model']['max_seq_length']
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['base_model'],
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=config['model']['load_in_4bit'],
    )

    print("2. Injecting LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        target_modules=config['lora']['target_modules'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        use_gradient_checkpointing=config['lora']['use_gradient_checkpointing'],
        random_state=config['training']['seed'],
        use_rslora=False,
        loftq_config=None,
    )

    print("3. Loading and Tokenizing Dataset...")
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    train_path = os.path.join(args.data_dir, "train.csv")
    train_df = pd.read_csv(train_path)
    train_df['conversations'] = train_df['conversations'].apply(json.loads)
    train_dataset = Dataset.from_pandas(train_df)

    def format_prompts(examples):
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                 for convo in examples["conversations"]]
        return {"text": texts}

    train_dataset = train_dataset.map(format_prompts, batched=True)

    print("4. Configuring SFTTrainer...")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        warmup_ratio=config['training']['warmup_ratio'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        logging_steps=config['training']['logging_steps'],
        optim=config['training']['optim'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        seed=config['training']['seed'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    print("5. Executing Training Phase...")
    resume_from_checkpoint = False
    if os.path.isdir(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        resume_from_checkpoint = True
        print(f"   -> Checkpoint detected at {checkpoint_dir}. Resuming training...")

    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("6. Saving Final Artifacts...")
    final_model_path = os.path.join(args.output_dir, "banking-intent-llama31-8b")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✅ Training Completed. Adapter securely persisted to: {final_model_path}")

if __name__ == "__main__":
    main()