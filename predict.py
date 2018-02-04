"""Prediction script.
"""
import importlib

import pandas as pd

import tools


# @tools.debug
def main(data_df=None,
         data_source=None,
         model_type=None,
         model_version=None):
    """Perform a prediction.
    Args:
        data_df (DataFrame): dataset to predict.
        data_source (str): folder of the dataset.
        model_type (str): type of model to use for prediction.
        model_version (str): version of this type of model to use.
    """
    # Load dataset if none were given
    if data_df is None:
        data_df = load_data(data_source)

    # Init the model
    model = init_model(model_type)

    # Load the model
    model.load_parameters(model_version)

    # Perform the prediction
    prediction_df = model.predict(data_df)

    # Return the table
    return prediction_df


def load_data(data_source):
    """Load dataset from the folder.
    Args:
        data_source (str): folder of the data
    Return:
        data_df (DataFrame): loaded dataset.
    """
    # Load the data
    data_df = pd.read_csv("data/{}/data.csv".format(data_source))

    # Return the dataset
    return data_df


def init_model(model_type):
    """Init the model.
    Args:
        model_type (str): type of the model to use.
    Return:
        model (Object): initialised model instance.
    """
    # Import the module
    model_module = importlib.import_module("library.{}.model".format(model_type))

    # Initialize an instance of the class
    model = model_module.Model()

    # Return the instance
    return model


if __name__ == '__main__':
    DATA_DF = pd.read_csv("data/us_election/data.csv").iloc[0:100, :-1]
    for model in ["scikit_learn_sag", "diy"]:
        main(data_df=DATA_DF,
             data_source=None,
             model_type=model,
             model_version="X")
