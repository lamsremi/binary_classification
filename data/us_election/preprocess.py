"""
Script to preprocess the us election dataset.

source: https://github.com/saimadhu-polamuri/DataAspirant_codes/tree/master/Logistic_Regression/Logistic_Binary_Classification

944 rows after removing wrong lines
"""
import pandas as pd

pd.set_option('display.width', 800)


def main():
    """
    Preprocess the data.
    """
    # Load the raw data
    raw_data_df = load_raw_data(path_raw_data="data/us_election/raw_data/data.csv")
    # Study data
    study_data(raw_data_df)
    # Transform the data
    data_df = process(raw_data_df)
    # Study transformed data
    study_data(data_df)
    # Store the data
    store(data_df, path_preprocessed_data="data/us_election/data.csv")


def load_raw_data(path_raw_data):
    """Load the raw data."""
    raw_data_df = pd.read_csv(
        path_raw_data,
    )
    return raw_data_df


def study_data(data_df):
    """
    Examine the data.
    """
    # Display shape
    print("- shape :\n{}\n".format(data_df.shape))
    # Display data dataframe (raws and columns)
    print("- dataframe :\n{}\n".format(data_df.head(10)))
    # Display types
    print("- types :\n{}\n".format(data_df.dtypes))
    # Missing values
    print("- missing values :\n{}\n".format(data_df.isnull().sum()))


def process(raw_data_df):
    """
    Process the data so it can be used by the mdoel
    """
    data_df = raw_data_df.copy()
    for attribute in raw_data_df.columns:
        data_df[attribute] = raw_data_df[attribute].astype(float)
    return data_df


def store(data_df, path_preprocessed_data):
    """Store the processed data."""
    data_df.to_csv(
        path_preprocessed_data,
        index=False
        )
