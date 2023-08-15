import json
import requests
import yaml
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import time


API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b"
TIME_INTERVAL = .2
BACKUP_EVERY = 50
NAME_NEW_COL = 'response_similarity'



def read_credential():
    with open('credentials.yaml', "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def query(payload, headers):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def get_status(df):
    return f"{sum(df[NAME_NEW_COL] != -1)}/{len(df)}"


def main(api_token, path_to_df):

    df = pd.read_csv(path_to_df, index_col=0)
    
    if NAME_NEW_COL not in df.columns:
        df[NAME_NEW_COL] = -1
    else:
        print(f"Restored from {get_status(df)}")

    headers = {"Authorization": f"Bearer {api_token}"}

    pbar = tqdm(
            enumerate(df.iterrows()), 
            desc=f"status: {get_status(df)} | last_backup: NaN",
            total = len(df),
        )

    for it, (idx, row) in pbar:
        if row[NAME_NEW_COL] < 0:
            data = query(
                {
                    "inputs": {
                        "source_sentence": row['prompts'],
                        "sentences": [row['responses']]
                    }
                },
                headers,
            )
            try:
                df.at[idx, NAME_NEW_COL] = data[0]
            except Exception as e:
                print(e)
                print(data)
                exit()
            
            time.sleep(TIME_INTERVAL)
        
        if sum(df[NAME_NEW_COL] == -1) == 0:
            print('\n[x] No more scores to get')
            break

        if it % BACKUP_EVERY == 0:
                df.to_csv(path_to_df)
                pbar.set_description(f"status: {get_status(df)} | last_backup: {it}it")


    df.to_csv(path_to_df)
    




if __name__ == "__main__":
    
    api_token = read_credential()['huggingface_hub']
    parser = ArgumentParser(description='Get config file.')


    parser.add_argument('-p', '--path', required=True, 
                        help='Path to the CSV file')
    
    args = parser.parse_args()
    path_to_df = args.path

    main(api_token, path_to_df)
    