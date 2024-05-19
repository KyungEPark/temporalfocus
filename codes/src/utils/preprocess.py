def berttopn(filename, n):
    import pandas as pd
    df = pd.read_pickle(filename)
    sorted_df = df.sort_values(by='score', ascending=False, inplace=False)
    final_df = sorted_df[0:n]
    return final_df

def liwctopn(filename, n):
    import pandas as pd
    
    # Read DataFrame from pickle file
    df = pd.read_pickle(filename)
    
    selected_rows = pd.DataFrame(columns=df.columns)
    labels = ['Past', 'Present', 'Future']
    
    for tense in labels:
        # Select rows for the current label
        tense_df = df[df['Label'] == tense]
        
        # Sort the selected rows by score
        sorted_tense = tense_df.sort_values(by='score', ascending=False)
        
        # Select top n rows for the current label
        semifinal = sorted_tense[0:n]
        
        selected_rows = pd.concat([selected_rows, semifinal])
    
    return selected_rows


def preprocess(df):
    texts = []
    for index, line in df.iterrows():
        text = f"The text: '{line['Text']}' has a temporal focus of {line['Label']} "
        texts.append(text)
    return texts
