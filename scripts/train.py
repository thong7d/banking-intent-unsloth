import os
import yaml
import json
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

def load_local_data(file_path):
    """
    Doc du lieu tu CSV va chuyen doi cot conversations tu chuoi JSON sang danh sach dict.
    """
    df = pd.read_csv(file_path)
    df['conversations'] = df['conversations'].apply(json.loads)
    return Dataset.from_pandas(df)

def main(config_path="configs/train.yaml"):
    # Doc cau hinh
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load Model va Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['name'],
        max_seq_length=config['model']['max_seq_length'],
        load_in_4bit=config['model']['load_in_4bit'],
    )

    # 2. Cau hinh PEFT (LoRA)
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        bias=config['lora']['bias'],
        use_gradient_checkpointing=config['lora']['use_gradient_checkpointing'],
        random_state=config['lora']['random_state'],
    )

    # 3. Thiet lap Chat Template cho Llama-3.1
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    # 4. Chuan bi du lieu
    train_dataset = load_local_data(config['data']['train_path'])
    val_dataset = load_local_data(config['data']['val_path'])

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    # 5. Khoi tao Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config['model']['max_seq_length'],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=TrainingArguments(
            output_dir=config['training']['output_dir'],
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
            eval_strategy=config['training']['eval_strategy'],
            save_strategy=config['training']['save_strategy'],
            report_to="none",
        ),
    )

    # 6. Ap dung train_on_responses_only
    # Trich xuat phan instruction va response dua tren template cua Llama-3.1
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # 7. Huan luyen
    print("Bat dau qua trinh huan luyen...")
    trainer.train()

    # 8. Luu model adapter
    model.save_pretrained(config['training']['output_dir'])
    tokenizer.save_pretrained(config['training']['output_dir'])
    print(f"Hoan tat huan luyen. Checkpoint da luu tai: {config['training']['output_dir']}")

if __name__ == "__main__":
    main()