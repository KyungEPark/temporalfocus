from src.eval import performance
from src.utils.fewshot import fewshot_tempfoc
from src.utils.preprocess import preprocess, randomselectn
from src.utils.hflogin import hflogin
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
import argparse
import json
import torch


def main(filename, n, labeled_file, output_file):
    hflogin()
    # Load data
    df = pd.read_pickle(filename)

    # Create an empty column 'Prediction'
    df['Prediction'] = None
    
 # Take random n examples from the handcoded one
    expath = r'data/rawdata/reasonedanno.pkl'
    exdf = randomselectn(expath, n)
    exex = preprocess(exdf)
    examples = '. '.join(exex)
    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token = access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    for index, row in tqdm(df.iterrows()):
        tempfoc = fewshot_tempfoc(model, tokenizer, row['Sentence'], examples)
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
    parser.add_argument("--n", type=int, required=True,
                        help="Number of examples used")
    parser.add_argument("--labeled_file", type=str, required=True,
                        help="Path to the labeled file")    
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the performance file")
    args = parser.parse_args()

    main(args.filename, args.n, args.labeled_file, args.output_file)