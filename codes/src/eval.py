from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def performance(df):
    # Calculate accuracy
    accuracy = accuracy_score(df['Label'], df['Prediction'])

    # Calculate precision, recall, and F1-score
    precision = precision_score(df['Label'], df['Prediction'], average='weighted', zero_division=1)
    recall = recall_score(df['Label'], df['Prediction'], average='weighted', zero_division=1)
    f1 = f1_score(df['Label'], df['Prediction'], average='weighted', zero_division=1)

    # Create a DataFrame to store the performance metrics
    performance_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
        'Value': [accuracy, precision, recall, f1]
    })

    return performance_df
