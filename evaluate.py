"""Module to evaluate the performance of a model.
"""
import pandas as pd
import numpy as np

import predict
import tools

from performance.numerical_bench import confusion_matrix

# @tools.debug
def main(model_type,
         model_version,
         dataset):
    """Perform test.
    """
    # Load labeled data
    inputs_data, truth_data = load_data(dataset)

    # Predict
    prediction_data = predict.main(inputs_data=inputs_data,
                                   model_type=model_type,
                                   model_version=model_version)

    # Quantitative performance
    performance = evaluate(truth_data, prediction_data)


def load_data(dataset):
    """Load traiing data.
    Args:
        dataset (str): source to take the data from.
    Return:
        inputs_data (DataFrame): loaded table of input data.
        truth_data (DataFrame): expected output.
    """
    # Load data
    data_df = pd.read_csv("data/{}/data.csv".format(dataset),
                          nrows=400)
    # Inputs data
    inputs_data = data_df.iloc[:, :-1]
    # Truth
    truth_data = data_df.iloc[:, -1]
    # Return the table
    return inputs_data, truth_data



@tools.debug
def evaluate(truth_data, prediction_data):
    """Computethe numerical performance."""
    # Set test
    y_truth = np.array(truth_data)

    # Set prediction
    y_prediction = np.array(prediction_data)

    # Compute confusion matrix
    result = confusion_matrix(y_truth, y_prediction)

    # Return result
    return result


if __name__ == '__main__':
    for model in ["scikit_learn_sag", "diy"]:
        main(model_type=model,
             model_version="X",
             dataset="us_election")
