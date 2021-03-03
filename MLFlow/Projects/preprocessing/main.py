import click
import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zipfile import ZipFile


def extract_data(path="../data", zip_name="titanic.zip", data_name="train.csv"):
    """Extract dataset.
        Parameters:
            path (str): Path to extract dataset without ending with slash
            zip_name (str): Name of the zip file to extract
            data_name (str): Name of one of the csv file to extract
    """

    if not os.path.exists(f"{path}/{data_name}"):
        ZipFile(f"{path}/{zip_name}", mode='r').extractall(path=path, members=None, pwd=None)


def preprocess_data(train_set):
    """Preprocess dataset.\n
    It will normalise numeric values and transorm to float non numeric values
        Parameters:
            train_set (pd.dataframe): pandas datafram containing all the data
    """

    train_set.loc[0:len(train_set) - 1, 'Pclass':'Pclass'] = [np.nanmean(train_set["Pclass"].values) if np.isnan(value) else value - 2 for value in train_set["Pclass"].values]
    train_set.loc[0:len(train_set) - 1, 'Sex':'Sex'] = [1 if value == "female" else -1 for value in train_set["Sex"].values]
    train_set.loc[0:len(train_set) - 1, 'Age':'Age'] = [np.nanmean(train_set["Age"].values) if np.isnan(value) else value / 10 for value in train_set["Age"].values]
    train_set.loc[0:len(train_set) - 1, 'Fare':'Fare'] = [np.nanmean(train_set["Fare"].values) if np.isnan(value) else value / 10 for value in train_set["Fare"].values]
    train_set.loc[0:len(train_set) - 1, 'Embarked':'Embarked'] = [-1 if value == 'Q' else 0 if value == 'S' else 1 for value in train_set["Embarked"].values]

    return train_set


def create_data_viz(train_set):
    """Create data viz plot.\n
    Compare Surviving rate between men and women
        Parameters:
            train_set (pd.dataframe): pandas datafram containing all the data
    """
    f, axes = plt.subplots(1, 3)

    axes[0].set_xlabel('Surviving rate of women')
    axes[1].set_xlabel('Global surviving rate')
    axes[2].set_xlabel('Surviving rate of men')

    sns.barplot(data=train_set[train_set['Sex'] == -1]['Survived'].values, estimator=lambda x: sum(x==0)*100.0/len(x), ax=axes[0], color="r").set(ylim=(0, 100))
    sns.barplot(data=train_set['Survived'].values, estimator=lambda x: sum(x==0)*100.0/len(x), ax=axes[1], color="g").set(ylim=(0, 100))
    sns.barplot(data=train_set[train_set['Sex'] == 1]['Survived'].values, estimator=lambda x: sum(x==0)*100.0/len(x), ax=axes[2], color="b").set(ylim=(0, 100))

    if not os.path.exists("artifacts"): os.mkdir("artifacts")
    plt.savefig("artifacts/plot.png")

@click.command(
    help="Given the titanic zip file (see dataset_folder and zip_name), exctract it"
    ", preprocess it and save it as npy file"
)
@click.option("--dataset-folder", default="../data", type=str)
@click.option("--zip-name", default="titanic.csv", type=str)
@click.option("--data-name", default="train.csv", type=str)
def preprocess(dataset_folder, zip_name, data_name):
    # path = "../data/train.csv" if len(sys.argv) < 2 else sys.argv[1]
    # zip_name = "../data/train.csv" if len(sys.argv) < 3 else sys.argv[2]
    # data_name = "../data/train.csv" if len(sys.argv) < 4 else sys.argv[3]

    extract_data(path=dataset_folder, zip_name=zip_name, data_name=data_name)

    train_set = pd.read_csv(f"{dataset_folder}/{data_name}", low_memory=False).drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    train_set = preprocess_data(train_set)

    create_data_viz(train_set)

    train_labels = train_set['Survived']
    train_set = train_set.drop('Survived', axis=1)

    np.save("../data/train_set.npy", train_set)
    np.save("../data/train_labels.npy", train_labels)

if __name__ == "__main__":
    preprocess()