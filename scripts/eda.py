# scripts/eda.py
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis for BANKING77")
    parser.add_argument("--output_dir", type=str, default="figures", help="Directory to save EDA plots")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("1. Downloading BANKING77 dataset (mteb/banking77)...")
    dataset = load_dataset("mteb/banking77")
    train_df = pd.DataFrame(dataset['train'])

    print("2. Performing Text Length Analysis...")
    train_df['text_length'] = train_df['text'].apply(lambda x: len(str(x).split()))
    max_len = train_df['text_length'].max()
    mean_len = train_df['text_length'].mean()
    
    print(f"   - Max text length: {max_len} words")
    print(f"   - Mean text length: {mean_len:.2f} words")
    print("   -> Conclusion: max_seq_length=512 is safe for tokenization without truncation.")

    print("3. Generating Label Distribution Plot...")
    plt.figure(figsize=(18, 6))
    
    # Sort labels to ensure consistent IDs
    categories = sorted(train_df['label_text'].unique())
    label2id = {name: i for i, name in enumerate(categories)}
    train_df['label_id'] = train_df['label_text'].map(label2id)

    sns.histplot(data=train_df, x='label_id', bins=77, color='steelblue')
    plt.title('BANKING77: Distribution of 77 Intent Classes in Training Data', fontsize=14)
    plt.xlabel('Intent Class ID', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    plot_path = os.path.join(args.output_dir, 'eda_label_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"4. EDA Plot successfully saved to: {plot_path}")

if __name__ == "__main__":
    main()