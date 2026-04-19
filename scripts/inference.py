import yaml
import json
import os
import torch
import argparse
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, config_path, checkpoint_dir=None, data_dir=None):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        model_path = checkpoint_dir if checkpoint_dir else self.config['model']['checkpoint_path']
        actual_data_dir = data_dir if data_dir else self.config['inference']['data_dir']

        print(f"Loading Model from: {model_path}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.config['model'].get('max_seq_length', 512),
            load_in_4bit=self.config['model'].get('load_in_4bit', True),
        )
        FastLanguageModel.for_inference(self.model)

        label_path = os.path.join(actual_data_dir, "id2label.json")
        with open(label_path, 'r', encoding='utf-8') as f:
            self.id2label = json.load(f)
            
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.system_prompt = self.config['inference']['system_prompt']

    def __call__(self, message):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f'Classify the intent of this banking message:\n\n"{message}"'}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs, max_new_tokens=32, temperature=0.1, do_sample=False)
        
        pred = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

        if pred not in self.label2id:
            for valid_label in self.label2id:
                if pred.lower() in valid_label.lower() or valid_label.lower() in pred.lower():
                    return valid_label
        return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()

    classifier = IntentClassification(args.config, args.checkpoint_dir, args.data_dir)
    
    test_queries = [
        "I am still waiting on my card?",
        "How do I change my PIN?",
        "There's a transaction I don't recognize",
    ]
    
    print("\n--- Testing Inference ---")
    for q in test_queries:
        print(f"Query : {q}")
        print(f"Intent: {classifier(q)}\n")