from src.train import llama3_unsloth
from src.utils.finetune_preprocess import makeexamples, formatting_prompts_func
from transformers import TrainingArguments
from src.utils.hflogin import hflogin
import pandas as pd
from tqdm import tqdm
import argparse
import json
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from datasets import load_dataset
from src.utils.finetune_preprocess import makeexamples, formatting_prompts_func

# Make sure to adjust the args before running

def main(bertn, liwcn, output_path):
    hflogin()
    setting = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 60,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = "outputs",
            optim = "adamw_8bit",
            seed = 3407, 
            report_to="wandb"
        ),
    llama3_unsloth(bertn, liwcn, setting, output_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bertn", type=int, required=True,
                    help="Path to the train data file")
    parser.add_argument("--liwcn", type=int, required=True,
                    help="Path to the train data file")
    parser.add_argument("--output_path", type=str, required=True,
                    help="Path to the trained model file")
    args = parser.parse_args()

    main(args.bertn, args.liwcn, args.output_path)