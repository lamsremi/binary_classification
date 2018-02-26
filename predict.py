"""Prediction script.
"""
import importlib

import pandas as pd

import tools


# @tools.debug
def main(inputs_data=None,
         model_type=None,
         model_version=None):
    """Perform a prediction.
    Args:
        inputs_data (DataFrame): dataset to predict.
        model_type (str): type of model to use for prediction.
        model_version (str): version of this type of model to use.
    """
    # Init the model
    model = init_model(model_type)

    # Load the model
    model.load_parameters(model_version)

    # Perform the prediction
    outputs_data = model.predict(inputs_data)

    # Return the table
    return outputs_data


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
    INPUTS_DATA = pd.read_csv("data/us_election/data.csv").iloc[0:100, :-1]
    for model in ["scikit_learn_sag", "diy"]:
        main(inputs_data=INPUTS_DATA,
             data_source=None,
             model_type=model,
             model_version="X")
