"""
Script to preprocess the diabete dataset

source: https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names

For Each Attribute: (all numeric-valued)
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)

768 rows after removing wrong lines
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
        names=[
            "pregnancy_time", "glucose_concentration",
            "blood_pressure", "skin_thickness",
            "insulin_serum", "body_mass",
            "diabete_pedigree", "age", "class"]
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


if __name__ == '__main__':
    PATH_RAW_DATA = "raw_data/data.txt"
    PATH_PREPROCESSED_DATA = "data.csv"
    main(PATH_RAW_DATA, PATH_PREPROCESSED_DATA)