from src.utils.preprocess import berttopn, liwctopn, preprocess
import argparse
import json
import pandas as pd


def main(filename, n, output_file):
    # Load data
    if 'liwc' in filename.lower():
        df = liwctopn(filename, n)

    elif 'bert' in filename.lower():
        df = berttopn(filename, n)

    preprocessed_texts = preprocess(df)
    
    # Save preprocessed texts to a file
    with open(output_file, 'w') as f:
        json.dump(preprocessed_texts, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True,
                        help="Path to the data file")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of top records to select")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file")
    args = parser.parse_args()

    main(args.filename, args.n, args.output_file)