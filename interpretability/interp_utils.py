import pandas as pd



def _assign_label(value):
    if value < 0.333:
        return 'low'
    if value >= 0.333 and value <= 0.666:
        return 'mid'
    else:
        return 'high'
    

def stratify_df(df: pd.DataFrame, subset_size):
    assert 'pro_API_response_score' in df.columns
    
    df['label'] = df['pro_API_response_score'].apply(_assign_label)

    # selecting stratified
    c = df['label'].value_counts().apply(lambda x: x/len(df))
    stratified_df = pd.concat([
            group.sample(
                int(c[lbl] * subset_size), 
                replace=False, 
                random_state=42
            ) for lbl, group in df.groupby('label')
    ])
    return stratified_df