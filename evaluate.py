"""
Script for performance

- Select the model to assess the performance
"""
import pandas as pd
import numpy as np

import predict
import tools

from performance.numerical_bench import confusion_matrix

# @tools.debug
def main(model_type, model_version, data_source):
    """Perform test."""
    # Load labeled data
    data_df = load_data(data_source)

    # Predict
    prediction_df = predict.main(data_df=data_df.iloc[:, :-1],
                                 data_source=None,
                                 model_type=model_type,
                                 model_version=model_version)

    # Quantitative performance
    performance = evaluate(data_df, prediction_df)


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



@tools.debug
def evaluate(data_df, prediction_df):
    """Computethe numerical performance."""
    # Set test
    y_test = np.array(data_df.iloc[:, -1])

    # Set prediction
    y_prediction = np.array(prediction_df.iloc[:, 0])

    # Compute confusion matrix
    result = confusion_matrix(y_test, y_prediction)

    # Return result
    return result


if __name__ == '__main__':
    for model in ["scikit_learn_sag", "diy"]:
        main(model_type=model,
             model_version="X",
             data_source="us_election")
