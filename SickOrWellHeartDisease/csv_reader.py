from pandas import DataFrame, read_csv
import pandas as pd


def load_csv(filepath):
    df = pd.read_csv(filepath)
    return df
