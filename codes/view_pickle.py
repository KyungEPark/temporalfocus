import pandas as pd
import pickle

# Path to your pickle file

file_name = 'cot1exam4perf.pkl'
pickle_file_path = 'data/output/validation/'+file_name

'''
file_name = 'synthdata.pkl'
pickle_file_path = 'data/rawdata/'+file_name
'''
# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

print(file_name)

# Check if the loaded data is a DataFrame
if isinstance(data, pd.DataFrame):
    print("Data loaded as a DataFrame:")
    print(data.to_string())
    print(data.dtypes)
else:
    print("The loaded data is not a DataFrame.")

'''
from src.utils.preprocess import randomselectn, cotpreprocess
ncotex = randomselectn(pickle_file_path, 1)
cotex = cotpreprocess(ncotex)
print(ncotex)
print(cotex)
'''