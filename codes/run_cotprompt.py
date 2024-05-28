from src.eval import performance
from src.utils.fewshotreasoning import fewshot_reastempfoc
from src.utils.preprocess import randomselectn, cotpreprocess
from src.utils.hflogin import hflogin
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import argparse
import json
import re
import torch
import numpy as np

def main(filename, n, labeled_file, output_file):
    hflogin()
    # Load data
    df = pd.read_pickle(filename)

    # Create an empty column 'Prediction'
    df['Prediction'] = None
    
    # Take random n examples from the handcoded one
    cotpath = r'data/rawdata/reasonedanno.pkl'
    cotdf = randomselectn(cotpath, n)
    cotex = cotpreprocess(cotdf)
    if len(cotex) > 1:  # Check if there are multiple elements
        examples = '. '.join(cotex)
    else:
        examples = cotex[0]  # If there's only one element, just use it directly
    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    

    pattern = r"reason: (.+) answer: (.+)"

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        tempfoc = fewshot_reastempfoc(model, tokenizer, row['Sentence'], examples)
        # Assign value to the 'reasonedlabel' column in the DataFrame
        df.at[index, 'reasonedlabel'] = tempfoc
        tempfoc = tempfoc.replace('\n', ' ')
        match = re.search(pattern, tempfoc)
        if match:
            df.at[index, 'Reason'] = match.group(1).strip()
            if "past" in match.group(2).strip():
                df.at[index, 'Prediction'] = "Past"
            elif "present" in match.group(2).strip():
                df.at[index, 'Prediction'] = "Present"
            elif "future" in match.group(2).strip():
                df.at[index, 'Prediction'] = "Future"
            else:
                df.at[index, 'Prediction'] = "Not Defined"

        
    
    # Assess performance
    if df['Prediction'].isna().any():
        print("There are None values in the 'Prediction' column.")
        print(df[df['Prediction'].isna()])
    # Filter out rows where 'Prediction' is None before assessing performance
    df_filtered = df.dropna(subset=['Prediction'])
    perf = performance(df_filtered)
    print("Performance assessed successfully.")
    print(perf)

    # Save performance dataframe as pickle
    perf.to_pickle(output_file)
    df.to_pickle(labeled_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True,
                        help="Path to the data file")
    parser.add_argument("--n", type=int, required=True,
                        help="Number of examples used")
    parser.add_argument("--labeled_file", type=str, required=True,
                        help="Path to the labeled file")    
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the performance file")
    args = parser.parse_args()

    main(args.filename, args.n, args.labeled_file, args.output_file)