import pandas as pd

def add_rul(df):
    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    df = df.drop("max_cycle", axis=1)
    return df


def add_features(df):
    # Normalize cycle (important feature)
    df["cycle_norm"] = df["cycle"] / df.groupby("engine_id")["cycle"].transform("max")

    # Rolling mean for first few sensors
    for i in range(1, 6):
        col = f"sensor_{i}"
        if col in df.columns:
            df[f"{col}_rolling"] = df.groupby("engine_id")[col].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )

    return df