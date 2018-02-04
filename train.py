"""Module used for fitting a model given a training dataset.
"""
import importlib

import pandas as pd

import tools


def main(data_df=None,
         data_source=None,
         model_type=None,
         starting_version=None,
         stored_version=None):
    """Fit a model.
    Args:
        data_df (DataFrame): dataset use to train
        data_source (str): source to take the training dataset from.
        model_type (str): type of model to train.
    """
    # Load data if None given
    if data_df is None:
        data_df = load_data(data_source)

    # Init the model
    model = init_model(model_type)

    # Load parameters
    model.load_parameters(model_version=starting_version)

    # Fit the model
    model.fit(data_df,
              alpha=0.00001,
              epochs=20)

    # Persist the parameters
    model.persist_parameters(model_version=stored_version)

    # Return bool
    return True


def load_data(data_source):
    """Load traiing data.
    Args:
        data_source (str): source to take the data from.
    Return:
        data_df (DataFrame): loaded table.
    """
    # Load data
    data_df = pd.read_csv("data/{}/data.csv".format(data_source),
                          nrows=400)

    # Return the table
    return data_df


def init_model(model_type):
    """Init an instance of the model
    Args:
        model_type (str): type of model to train.
    Return:
        model (instance Object): init model.
    """
    # Import model
    model_module = importlib.import_module("library.{}.model".format(model_type))

    # Init
    model = model_module.Model()

    # Return model
    return model


if __name__ == '__main__':
    for source in ["us_election"]:
        for model in ["scikit_learn_sag", "diy"]:
            main(data_df=None,
                 data_source=source,
                 model_type=model,
                 starting_version=None,
                 stored_version="X")
