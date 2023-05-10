import pandas as pd

def download_DIALOCONAN():
    CSV_URL = 'https://raw.githubusercontent.com/marcoguerini/CONAN/master/DIALOCONAN/DIALOCONAN.csv'
    return pd.read_csv(CSV_URL)

