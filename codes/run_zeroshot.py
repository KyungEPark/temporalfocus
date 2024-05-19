from src.eval import performance
from src.utils.zeroshot import zeroshot_tempfoc
from src.utils.hflogin import hflogin
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import argparse
import json
import torch


def main(filename, labeled_file, output_file):
    hflogin()
    # Load data
    df = pd.read_pickle(filename)

    # Create an empty column 'Prediction'
    df['Prediction'] = None
    
    access_token = "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"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token = access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    

    for index, row in tqdm(df.iterrows()):
        tempfoc = zeroshot_tempfoc(model, tokenizer, row['Sentence'])
        # Assign value to the 'Prediction' column in the DataFrame
        df.at[index, 'Prediction'] = tempfoc
    # Assess performance
    perf = performance(df)

    # Save performance dataframe as pickle
    perf.to_pickle(output_file)
    df.to_pickle(labeled_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True,
                        help="Path to the data file")
    parser.add_argument("--labeled_file", type=str, required=True,
                        help="Path to the labeled file")    
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file")
    args = parser.parse_args()

    main(args.filename, args.labeled_file, args.output_file)