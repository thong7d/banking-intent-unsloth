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

    # Định tuyến đường dẫn động
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
        subprocess.run(["python", "scripts/eda.py", "--output_dir", figures_dir])

    elif args.step in ["preprocess", "all"]:
        subprocess.run(["python", "scripts/preprocess_data.py", "--output_dir", data_dir, "--seed", "42"])
        
    elif args.step in ["train", "all"]:
        subprocess.run(["python", "scripts/train.py", "--config", "configs/train.yaml", "--output_dir", output_dir, "--data_dir", data_dir])

    elif args.step == "evaluate":
        ft_model_path = os.path.join(output_dir, "banking-intent-llama31-8b")
        subprocess.run(["python", "scripts/evaluate.py", "--config", "configs/train.yaml", "--model_path", ft_model_path, "--data_dir", data_dir, "--output_dir", figures_dir])

    elif args.step in ["infer", "all"]:
        print("Executing Standalone Inference...")
        # Truyền đường dẫn Google Drive tuyệt đối xuống cho script
        checkpoint_dir = os.path.join(output_dir, "banking-intent-llama31-8b")
        subprocess.run(["python", "scripts/inference.py", 
                        "--config", "configs/inference.yaml", 
                        "--checkpoint_dir", checkpoint_dir, 
                        "--data_dir", data_dir])

if __name__ == "__main__":
    main()