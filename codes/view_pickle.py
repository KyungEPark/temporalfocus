import pandas as pd
import pickle

# Path to your pickle file
file_name = 'few1labeled.pkl'
pickle_file_path = 'data/output/validation/'+file_name

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

print(file_name)
# Check if the loaded data is a DataFrame
if isinstance(data, pd.DataFrame):
    print("Data loaded as a DataFrame:")
    print(data.to_string())
else:
    print("The loaded data is not a DataFrame.")
