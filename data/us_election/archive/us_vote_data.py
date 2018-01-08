"""
script to process the data set election

source: https://github.com/saimadhu-polamuri/DataAspirant_codes/tree/master/Logistic_Regression/Logistic_Binary_Classification

944 rows
"""
import pickle
import math

import numpy as np
import pandas as pd

import tools


pd.set_option('display.width', 800)

# @tools.debug
def process_data(path_csv, path_pickle):
    """
    Process the data.
    """
    # Load the data
    data_df = load_raw_data(path_csv)
    # Process the data into the adequate format
    data_array = convert_format(
        data_df)
    # Store the data
    store_data(data_array, path_pickle)


# @tools.debug
def load_raw_data(path_csv):
    """Load raw data."""
    data_df = pd.read_csv(
        path_csv,
        # nrows=4,
        dtype=np.float
        # names=[
        #     "pregnancy_time", "glucose_concentration",
        #     "blood_pressure", "skin_thickness",
        #     "insulin_serum", "body_mass",
        #     "diabete_pedigree", "age", "class"]
        )
    return data_df


# @tools.debug
def convert_format(data_df):
    """
    Convert into the adequate format.
    data_array = [
        [
            X_11,
            X_12,
            y_13
        ],
        [
            X_21,
            X_22,
            y_23
        ]
    ]
    """
    # data_array = np.array(data_df)
    data_raw_array = data_df.values.tolist()
    # Check that all values are dtype('float64')
    data_array = []
    for data in data_raw_array:
        if all(isinstance(item, np.float) and not math.isnan(item) for item in data):
            data_array.append(data)
    return data_array


def store_data(data_array, path_pickle):
    """Store the data_array as pickle."""
    pickle.dump(data_array, open(path_pickle, "wb"))


if __name__ == '__main__':
    process_data(path_csv="anes_dataset.csv", path_pickle="data_array.pkl")