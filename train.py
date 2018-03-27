"""Module to fit a model given a labeled dataset."""
import sys
import pickle
import importlib

import pandas as pd

import tools


def main(model_type,
         start_version,
         end_version,
         data_source,
         labeled_data):
    """Fit a model.
    Args:
        model_type (str): type of model to train.
        start_version (str): version of model to start the training from.
        end_version (str): version to store the parameters.
        data_source (str): source to take the labeled dataset from.
        labeled_data (list): dataset use to train
        [
            ([ 0.0, 6.0, 6.0, 2.0], 1.0)
            ([ 8.0, 7.0, 4.0, 2.0], 1.0)
        ]
    """
    # Load data
    labeled_data = load_data(data_source) if labeled_data is None else labeled_data

    # Init the model
    model = init_model(model_type)

    # Load parameters
    model.load_parameters(model_version=start_version)

    # Fit the model
    model.fit(labeled_data,
              alpha=0.00001,
              epochs=20)

    # Persist the parameters
    model.persist_parameters(model_version=end_version)


def load_data(data_source):
    """Load labeled data."""
    with open("data/{}/data.pkl".format(data_source), "rb") as handle:
        labeled_data = pickle.load(handle)
    return labeled_data


def init_model(model_type):
    """Instanciate an instance of the class of the given type of model.
    Args:
        model_type (str): type of model to train.
    Return:
        model (Model): instance of Model.
    """
    # Import model
    model_module = importlib.import_module("library.{}.model".format(model_type))
    # Return the instance
    return model_module.Model()


if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else "pure_python"
    source = sys.argv[2] if len(sys.argv) > 2 else "us_election"
    main(model,
         start_version=None,
         end_version="X",
         data_source=source,
         labeled_data=None)
