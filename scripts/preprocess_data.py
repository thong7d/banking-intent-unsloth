"""
Banking Intent Classification - Data Preprocessing
========================================================
Thực thi các Phase 1-6: Sampling, Normalization, Formatting (ChatML) & Splitting.
"""

import os
import re
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Cấu hình tĩnh từ tài liệu
SAMPLES_PER_CLASS = 80
TEST_SIZE_VAL = 0.15
RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "You are a banking customer service intent classifier. "
    "Given a customer message, classify it into exactly one of the 77 banking intent categories. "
    "Respond with ONLY the intent label name, nothing else."
)

def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản: lowercase, xóa khoảng trắng thừa, giữ lại số và ký tự tiền tệ."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Giữ lại chữ, số, khoảng trắng và các ký tự: ' - ? . , ! $ € £ %
    text = re.sub(r'[^\w\s\'\-\?\.\,\!\$\€\£\%]', '', text)
    return text

def format_chatml(text: str, intent_name: str) -> str:
    """Đóng gói dữ liệu thành chuỗi JSON chuẩn ChatML cho Unsloth SFT."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Classify the intent of this banking message:\n\n"{text}"'},
        {"role": "assistant", "content": intent_name}
    ]
    return json.dumps(messages)

def main():
    print("🚀 Bắt đầu tiền xử lý dữ liệu BANKING77...")
    os.makedirs("sample_data", exist_ok=True)
    
    # 1. Load Dataset
    print("📦 Đang tải dataset PolyAI/banking77...")
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # 2. Xây dựng mapping id2label
    label_names = dataset['train'].features['label'].names
    id2label = {i: name for i, name in enumerate(label_names)}
    
    # 3. Stratified Sampling (Tập Train)
    print(f"⚖️ Lấy mẫu phân tầng: {SAMPLES_PER_CLASS} samples/class...")
    train_sampled = train_df.groupby('label').apply(
        lambda x: x.sample(n=min(SAMPLES_PER_CLASS, len(x)), random_state=RANDOM_SEED)
    ).reset_index(drop=True)
    
    # 4. Tiền xử lý & Ánh xạ nhãn
    print("🧹 Đang chuẩn hóa văn bản và ánh xạ nhãn...")
    for df in [train_sampled, test_df]:
        df['text_clean'] = df['text'].apply(normalize_text)
        df['intent_name'] = df['label'].map(id2label)
        # 5. Format sang ChatML
        df['conversations'] = df.apply(
            lambda row: format_chatml(row['text_clean'], row['intent_name']), axis=1
        )
    
    # 6. Chia tách Train/Validation
    print(f"🔀 Đang chia tập Train/Val (Tỷ lệ Validation: {TEST_SIZE_VAL * 100}%)...")
    train_split, val_split = train_test_split(
        train_sampled,
        test_size=TEST_SIZE_VAL,
        random_state=RANDOM_SEED,
        stratify=train_sampled['label']
    )
    
    # 7. Lưu trữ dữ liệu đầu ra
    columns_to_save = ['text_clean', 'label', 'intent_name', 'conversations']
    rename_mapping = {'text_clean': 'text'}
    
    train_split[columns_to_save].rename(columns=rename_mapping).to_csv('sample_data/train.csv', index=False)
    val_split[columns_to_save].rename(columns=rename_mapping).to_csv('sample_data/val.csv', index=False)
    test_df[columns_to_save].rename(columns=rename_mapping).to_csv('sample_data/test.csv', index=False)
    
    # Lưu dictionary mapping để dùng cho Phase Inference
    with open('sample_data/id2label.json', 'w', encoding='utf-8') as f:
        json.dump(id2label, f, indent=4)
        
    print(f"✅ Hoàn tất! Dữ liệu đã được lưu tại thư mục 'sample_data/':")
    print(f"   - Train: {len(train_split)} mẫu")
    print(f"   - Val:   {len(val_split)} mẫu")
    print(f"   - Test:  {len(test_df)} mẫu")

if __name__ == "__main__":
    main()