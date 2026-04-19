# main.py
import argparse
import subprocess
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Banking Intent Pipeline Orchestrator")
    parser.add_argument("--step", choices=["eda", "preprocess", "train", "evaluate", "infer", "all"], required=True)
    parser.add_argument("--env", choices=["local", "colab"], default="local")
    return parser.parse_args()

def main():
    args = parse_args()

    # Dynamic pathing based on environment
    if args.env == "colab":
        base_storage_dir = "/content/drive/MyDrive/banking-intent-unsloth"
    else:
        base_storage_dir = os.path.abspath(os.path.dirname(__file__))

    data_dir = os.path.join(base_storage_dir, "sample_data")
    output_dir = os.path.join(base_storage_dir, "outputs")
    figures_dir = os.path.join(base_storage_dir, "figures")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if args.step in ["eda", "all"]:
        print("Executing Exploratory Data Analysis...")
        subprocess.run(["python", "scripts/eda.py", "--output_dir", figures_dir])

    elif args.step in ["preprocess", "all"]:
        print("Executing Data Preprocessing...")
        subprocess.run(["python", "scripts/preprocess_data.py", "--output_dir", data_dir, "--seed", "42"])
        
    elif args.step in ["train", "all"]:
        print("Executing Model Training...")
        subprocess.run(["python", "scripts/train.py", "--config", "configs/train.yaml", "--output_dir", output_dir, "--data_dir", data_dir])
    elif args.step == "evaluate":
        print("Executing Model Evaluation...")
        # Đường dẫn tới adapter đã train ở Phase 8
        ft_model_path = os.path.join(output_dir, "banking-intent-llama31-8b")
        # Đường dẫn tới mô hình gốc để so sánh baseline
        base_model_path = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit"
        
        figures_dir = os.path.join(base_storage_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Đánh giá Fine-tuned
        print("Evaluating Fine-tuned Model...")
        subprocess.run(["python", "scripts/evaluate.py", 
                        "--config", "configs/train.yaml", 
                        "--model_path", ft_model_path, 
                        "--data_dir", data_dir, 
                        "--output_dir", figures_dir])
    elif args.step in ["infer", "all"]:
        print("Executing Standalone Inference...")
        subprocess.run(["python", "scripts/inference.py", "--config", "configs/inference.yaml"])

if __name__ == "__main__":
    main()