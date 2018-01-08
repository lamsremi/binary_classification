"""
Script for performance

- Select the model to assess the performance
"""
import pandas as pd
import numpy as np
from main import predict

import utils
import tools
from performance.numerical_bench import confusion_matrix

# @tools.debug
def evaluate(model_type, input_source):
    """Perform test."""
    # Load input
    x_test, y_test = load_input(input_source)
    # Predict output
    y_prediction, model = predict(
        x_test,
        model_type)
    # Quantitative performance
    performance = evaluate(y_test, y_prediction)
    # Qualitative performance [Graphical]
    # display_results(x_test, y_test, y_prediction)


# @tools.debug
def load_input(input_source):
    """Load the input based on the input source."""
    x_test, y_test = utils.load_data(
        "data/{}/data_array.pkl".format(input_source))
    # x_test, y_test = utils.filter_array(x_test, y_test, 0, 300)
    return x_test, y_test


@tools.debug
def evaluate(y_test, y_prediction):
    """Computethe numerical performance."""
    # Compute confusion matrix
    result = confusion_matrix(y_test, y_prediction)
    performance = []
    # for i_index in range(len(y_test)):
    #     performance.append([y_test[i_index], y_prediction[i_index]])
    return result


def display_results(x_test, y_test, y_prediction):
    """Display the results."""
    print("ROC curve to be coded......")


if __name__ == '__main__':
    MODEL_TYPE = "diy"
    INPUT_SOURCE = "us_election"
    evaluate(MODEL_TYPE, INPUT_SOURCE)
