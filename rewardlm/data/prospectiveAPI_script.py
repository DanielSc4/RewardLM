import yaml
import json
import pandas as pd
from tqdm import tqdm
import time
from googleapiclient import discovery
from argparse import ArgumentParser

ATTRIBUTE = 'TOXICITY'
NAME_NEW_COL = 'pro_API_response_score'
TIME_INTERVAL = 0.8      # prospective API limit the QPS (query per second) to 1 # https://console.cloud.google.com/iam-admin/quotas to check quota alerts
TRIALS = 4      # number of times to iterate on the entire dataset
BACKUP_EVERY = 25


def read_credential():
    with open('credentials.yaml', "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    
    return cfg['google_cloud']

def get_status(df):
    return f"{sum(df[NAME_NEW_COL] != -1)}/{len(df)}"

def main(path_to_df):
    API_KEY = read_credential()
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )
    

    
    df = pd.read_csv(path_to_df, index_col=0)
    # create column w/ -1 scores as default
    if NAME_NEW_COL not in df.columns:
        df[NAME_NEW_COL] = -1

    consecutive_err = 0
    count = 0       # total values obtained during execution


    print(f"Restored from {get_status(df)}")
    n_response = 0

    for _ in range(TRIALS):
        pbar = pbar = tqdm(enumerate(df.iterrows()), desc=f"status: {get_status(df)} | from last backup: {n_response} | last_backup: NaN")
        
        for it, (idx, row) in pbar:
            # if score is not assigned:
            if row[NAME_NEW_COL] < 0:
                time.sleep(TIME_INTERVAL)
                analyze_request = {
                    'comment': { 'text': row['responses']},
                    'requestedAttributes': {ATTRIBUTE: {},}
                }

                # catch HTTP errors
                try:
                    response = client.comments().analyze(body=analyze_request).execute()
                    # assign score
                    df.at[idx, NAME_NEW_COL] = response['attributeScores'][ATTRIBUTE]['summaryScore']['value']
                    n_response += 1
                    count += 1
                    consecutive_err = 0     # reset the counter for no response in a row
                except Exception as e:
                    if 'LANGUAGE_NOT_SUPPORTED_BY_ATTRIBUTE' in str(e):
                        df.at[idx, NAME_NEW_COL] = pd.NA
                    # if HTTP error, pass and try next time
                    pass
            
            # exit loop 
            if sum(df[NAME_NEW_COL] == -1) == 0:
                print('\n[x] No more scores to get')
                break
            
            # backup, overwrite
            if it % BACKUP_EVERY == 0:
                df.to_csv(path_to_df)
                pbar.set_description(f"status: {get_status(df)} | from last backup: {n_response}/{BACKUP_EVERY} | last_backup: {it}it")
                n_response = 0


        print(f'total scores obtained this loop: {count}')
        print()
    
    print(f'total scores in df: {get_status(df)}')
    print(f'total valid scores in df: {sum(df[NAME_NEW_COL] > 0)}/{len(df)}')
    # save csv
    df.to_csv(path_to_df)




if __name__ == "__main__":
    
    parser = ArgumentParser(description='Get config file.')
    parser.add_argument('-p', '--path', required=True, 
                        help='Path to the CSV file')
    args = parser.parse_args()
    path_to_df = args.path


    main(path_to_df)