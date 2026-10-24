import pandas as pd

#ex results = load_results("data/raw/results.csv")

def load_results(path) :
    return pd.read_csv(path)
