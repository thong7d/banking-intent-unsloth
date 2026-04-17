# scripts/preprocess_data.py
import os
import re
import json
import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocessing Pipeline for BANKING77")
    parser.add_argument("--output_dir", type=str, default="sample_data", help="Directory to save processed data")
    parser.add_argument("--samples_per_class", type=int, default=80, help="Number of samples per intent class")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation set split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\'\-\?\.\,\!\$\€\£\%]', '', text)
    return text

def format_chatml(text, intent_name):
    system_prompt = (
        "You are a banking customer service intent classifier. "
        "Given a customer message, classify it into exactly one of the 77 banking intent categories. "
        "Respond with ONLY the intent label name, nothing else."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Classify the intent of this banking message:\n\n"{text}"'},
        {"role": "assistant", "content": intent_name}
    ]
    return json.dumps(messages)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("1. Loading dataset (mteb/banking77)...")
    dataset = load_dataset("mteb/banking77")
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Build secure label mapping
    categories = sorted(list(set(test_df['label_text'])))
    id2label = {i: name for i, name in enumerate(categories)}
    label2id = {name: i for i, name in id2label.items()}
    
    train_df['label'] = train_df['label_text'].map(label2id)
    test_df['label'] = test_df['label_text'].map(label2id)
    
    print("2. Phase 4.1 - Deduplication...")
    initial_len = len(train_df)
    train_df = train_df.drop_duplicates(subset=['text'], keep='first')
    print(f"   - Removed {initial_len - len(train_df)} duplicate samples.")
    
    print(f"3. Phase 3 - Stratified Sampling (Target: {args.samples_per_class} per class)...")
    # Tắt cảnh báo deprecated của Pandas bằng include_groups=False
    train_sampled = train_df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(args.samples_per_class, len(x)), random_state=args.seed),
        include_groups=False
    ).reset_index(drop=True)
    
    # Restore 'label' column which is dropped when include_groups=False
    train_sampled['label'] = train_sampled['label_text'].map(label2id)
    print(f"   - Total training samples after sampling: {len(train_sampled)}")
    
    print("4. Phase 4.2 & 5 - Normalization and ChatML Formatting...")
    for df in [train_sampled, test_df]:
        df['text_clean'] = df['text'].apply(normalize_text)
        df['intent_name'] = df['label'].map(id2label)
        df['conversations'] = df.apply(
            lambda row: format_chatml(row['text_clean'], row['intent_name']), axis=1
        )
    
    print(f"5. Phase 6 - Train/Validation Split (Val Ratio: {args.val_size})...")
    train_split, val_split = train_test_split(
        train_sampled,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_sampled['label']
    )
    
    print("6. Saving final artifacts to persistent storage...")
    columns_to_save = ['text_clean', 'label', 'intent_name', 'conversations']
    rename_dict = {'text_clean': 'text'}
    
    train_split[columns_to_save].rename(columns=rename_dict).to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    val_split[columns_to_save].rename(columns=rename_dict).to_csv(os.path.join(args.output_dir, 'val.csv'), index=False)
    test_df[columns_to_save].rename(columns=rename_dict).to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
    
    with open(os.path.join(args.output_dir, 'id2label.json'), 'w', encoding='utf-8') as f:
        json.dump(id2label, f, indent=4)
        
    print(f"   - Output Directory: {args.output_dir}")
    print(f"   - Train Size: {len(train_split)} | Val Size: {len(val_split)} | Test Size: {len(test_df)}")
    print("✅ Data Preprocessing Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()