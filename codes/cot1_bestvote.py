import pandas as pd
from src.eval import performance
from functools import reduce

# Call for each example the predictions
for num in range(0, 10):
    try:
        path = f"/home/ma/ma_ma/ma_kyupark/is809/data/output/validation/cot1exam{num}labeled.pkl"
        globals()[f'cot{num}'] = pd.read_pickle(path)
        globals()[f'cot{num}'] = globals()[f'cot{num}'][["Sentence", "Prediction"]]
        globals()[f'cot{num}'] = globals()[f'cot{num}'].rename(columns={"Prediction": f"{num}Prediction"})
        
    except Exception as e:
        print(f"Failed for CoT {num}: {e}")
print(cot0.head())
print(cot1.head()) 

# Answer DataFrame
answer = pd.read_pickle(r"/home/ma/ma_ma/ma_kyupark/is809/data/output/validation/cot1exam0labeled.pkl")
answer = answer[["Sentence", "Label"]]

# Merge them into one DataFrame
dfs = [cot0, cot1, cot2, cot3, cot4, cot5, cot6, cot7, cot8, cot9]

# Merge using "reduce"
df = reduce(lambda left, right: pd.merge(left, right, on='Sentence'), dfs)

# Decide on the biggest vote
def most_common_prediction(row):
    predictions = row[1:].values 
    if len(predictions) == 0:
        return None
    try:
        return pd.Series(predictions).mode().iloc[0]
    except IndexError:
        return None 
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


# Select the most voted and apply to "Prediction"
df['Prediction'] = df.apply(most_common_prediction, axis=1)

# Merge with answer 
df = pd.merge(df, answer, on='Sentence')

print(df[1:10])

# Calculate performance
perf = performance(df)
print(perf)
perf.to_pickle(r"/home/ma/ma_ma/ma_kyupark/is809/data/output/validation/mostvoteperf.pkl")
