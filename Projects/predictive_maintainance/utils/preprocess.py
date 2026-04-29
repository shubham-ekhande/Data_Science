import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis=1)
    return df

def add_column_names(df):
    num_cols = df.shape[1]
    cols = ["engine_id", "cycle"] + [f"sensor_{i}" for i in range(1, num_cols - 1)]
    df.columns = cols
    return df