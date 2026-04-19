import argparse
import yaml
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--is_baseline", action="store_true", help="Evaluate baseline model without LoRA")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Load Model & Tokenizer
    print(f"Loading {'Baseline' if args.is_baseline else 'Fine-tuned'} model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=config['model']['max_seq_length'],
        load_in_4bit=config['model']['load_in_4bit'],
    )
    FastLanguageModel.for_inference(model)

    # 2. Load Label Mapping
    with open(os.path.join(args.data_dir, "id2label.json"), 'r') as f:
        id2label = json.load(f)
    
    # 3. Load Test Data
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    
    # 4. Inference loop
    print("Executing batch inference on test set...")
    predictions = []
    ground_truth = test_df['intent_name'].tolist()

    system_prompt = "You are a banking customer service intent classifier. Given a customer message, classify it into exactly one of the 77 banking intent categories. Respond with ONLY the intent label name, nothing else."

    for message in tqdm(test_df['text'].tolist()):
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Classify the intent of this banking message:\n\n"{message}"'}
        ], tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(input_ids=prompt, max_new_tokens=32, use_cache=True, temperature=0.1)
        
        pred = tokenizer.decode(outputs[0][prompt.shape[-1]:], skip_special_tokens=True).strip()
        predictions.append(pred)

    # 5. Calculate Metrics
    acc = accuracy_score(ground_truth, predictions)
    print(f"\n--- Evaluation Results ({'Baseline' if args.is_baseline else 'Fine-tuned'}) ---")
    print(f"Accuracy: {acc:.4f}")
    
    report = classification_report(ground_truth, predictions)
    report_path = os.path.join(args.output_dir, f"report_{'base' if args.is_baseline else 'finetuned'}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # 6. Confusion Matrix (Top 20 most confused classes for clarity)
    cm = confusion_matrix(ground_truth, predictions)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm[:20, :20], annot=True, fmt='d', cmap='Blues') # Show only top 20 for visibility
    plt.title(f"Confusion Matrix (Partial) - {'Baseline' if args.is_baseline else 'Fine-tuned'}")
    plt.savefig(os.path.join(args.output_dir, f"cm_{'base' if args.is_baseline else 'finetuned'}.png"))
    
    print(f"Artifacts saved to {args.output_dir}")

if __name__ == "__main__":
    main()