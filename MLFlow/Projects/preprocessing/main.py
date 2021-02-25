import os
import wget
import pandas as pd
from zipfile import ZipFile

def extract_data(path="../data", data_name="ratings.csv"):
    """Extract dataset.
        Parameters:
            path (str): Path to extract dataset without ending with slash
            data_name (str): Name of one of the csv file to extract
    """

    if not os.path.exists(f"{path}/{data_name}"):
        ZipFile(f"{path}/{data_name}", mode='r').extractall(path=path, members=None, pwd=None)

def select_useful_data(path="../data/movies_metadata.csv", cols=["belongs_to_collection", "budget", "genres", "popularity", "production_companies", "revenue", "vote_average"], label_colname="vote_average"):
    data = pd.read_csv(path, usecols=cols, low_memory=False)
    return data.drop(columns=[label_colname]), data[label_colname]

if __name__ == "__main__":
    extract_data()
    select_useful_data()