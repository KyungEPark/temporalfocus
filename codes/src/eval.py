from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def performance(df):

    df['Prediction'] = df['Prediction'].replace({None: pd.NA})

    # Drop the NaNs
    df = df.dropna(subset=['Prediction'])

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

def analyze_predictions(df):
    # Calculate class distribution
    class_distribution = df['Label'].value_counts()
    
    # Calculate confusion matrix
    cm = confusion_matrix(df['Label'], df['Prediction'])
    
    # Calculate performance metrics
    performance_df = performance(df)
    
    return performance_df, class_distribution, cm