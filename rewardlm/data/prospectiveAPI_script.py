import yaml
import json
import pandas as pd
from tqdm import tqdm
import time
from googleapiclient import discovery
from argparse import ArgumentParser


ATTRIBUTE = 'TOXICITY'
NAME_NEW_COL = 'pro_API_response_score'
TIME_INTERVAL = .95      # prospective API limit the QPS (query per second) to 1
# https://console.cloud.google.com/iam-admin/quotas to check quota alerts


def read_credential():
    with open('credentials.yaml', "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    
    return cfg['google_cloud']



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
    df[NAME_NEW_COL] = -1

    consecutive_err = 0
    count = 0       # total values obtained during execution

    pbar = pbar = tqdm(enumerate(df.iterrows()), desc='status')
    for it, (idx, row) in pbar:
        n_response = 0          # total values obtained during this cycle
        # if score is not assigned:
        if row[NAME_NEW_COL] < 0:
            time.sleep(TIME_INTERVAL)
            analyze_request = {
                'comment': { 'text': row['responses']},
                'requestedAttributes': {ATTRIBUTE: {},}
            }
            # prevent HTTP errors
            try:
                response = client.comments().analyze(body=analyze_request).execute()
                # assign score
                df.at[idx, NAME_NEW_COL] = response['attributeScores'][ATTRIBUTE]['summaryScore']['value']
                n_response += 1
                count += 1
                consecutive_err = 0     # reset the counter for no response in a row
            except:
                # if HTTP error, pass and try next time
                pass
        
        # exit loop 
        if sum(df[NAME_NEW_COL] == -1) == 0:
            # no more scores to get
            break

        # if, for more than 5 times in a row, the server does not provide any response, pause for 5 seconds
        # if, for more than 20 times in a row, the server does not provide any response, quit
        if n_response == 0:     # no responses obtained during this cycle
            consecutive_err += 1

            if consecutive_err > 5:
                time.sleep(4)

                # backup
                df.to_csv(
                    '{original_path_and_name}_w_pAPI.csv'.format(original_path_and_name = path_to_df.split('.csv')[0])
                )
                pbar.set_description(f"{sum(df[NAME_NEW_COL] != -1)}/{len(df)}, last_backup: {it}")

            if consecutive_err > 20:
                print('\nServer is not providing any response, 5 time in a row. Bye!')
                print(f'Num of missing values: {sum(df[NAME_NEW_COL] == -1)}/{len(df)}')
                break

    print(f'total scores obtained this time: {count}')
    print(f'total valid scores in df: {sum(df[NAME_NEW_COL] != -1)}/{len(df)}')
    # save csv
    df.to_csv(
        '{original_path_and_name}_w_pAPI.csv'.format(original_path_and_name = path_to_df.split('.csv')[0])
    )





if __name__ == "__main__":
    
    parser = ArgumentParser(description='Get config file.')
    parser.add_argument('-p', '--path', required=True, 
                        help='Path to the CSV file')
    args = parser.parse_args()
    path_to_df = args.path

    main(path_to_df)