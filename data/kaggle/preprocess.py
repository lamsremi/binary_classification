"""
Script to preprocess the kaggle titanic dataset.
"""
import pandas as pd

pd.set_option('display.width', 800)

def main(path_raw_data, path_preprocessed_data=None):
    """
    Preprocess the data.
    """
    # Load the raw data
    raw_data_df = load_raw_data(path_raw_data)
    # Study data
    study_data(raw_data_df)
    # Transform the data
    data_df = process(raw_data_df)
    # Study transformed data
    study_data(data_df)
    # Store the data
    store(data_df, path_preprocessed_data)


def load_raw_data(path_raw_data):
    """Load the raw data."""
    raw_data_df = pd.read_csv(
        path_raw_data,
        nrows=10000,
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
    # Select a subset of columns
    data_df = raw_data_df[[
        "Pclass",
        "Fare",
        "Age",
        "SibSp",
        "Parch",
        "Survived"]]
    # Convert to dtype float
    for attribute in data_df.columns:
        data_df[attribute] = raw_data_df[attribute].astype(float)
    # Drop all the NaN value
    data_df.dropna(inplace=True)
    return data_df


def store(data_df, path_preprocessed_data):
    """Store the processed data."""
    data_df.to_csv(
        path_preprocessed_data,
        index=False
        )


if __name__ == '__main__':
    PATH_RAW_DATA = "raw_data/data.csv"
    PATH_PREPROCESSED_DATA = "data.csv"
    main(PATH_RAW_DATA, PATH_PREPROCESSED_DATA)