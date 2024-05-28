from src.eval import performance, analyze_predictions
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


file_name = 'few1labeled.pkl'
pickle_file_path = 'data/output/validation/'+file_name
df = pd.read_pickle(pickle_file_path)

perf = performance(df)
performance_df, class_distribution, cm = analyze_predictions(df)
print(performance_df)
print(cm)