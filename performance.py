"""
Script for performance

- Select the model to assess the performance
"""
import pandas as pd
import numpy as np
from main import predict
import utils
import tools


def test(model_string):
    """Perform test."""
    # Load input
    x_test, y_test = load_input("diabete")
    # Predict output
    y_prediction, model = predict(x_test, model_string)
    # Quantitative performance
    # performance = evaluate(y_test, y_prediction)
    # Cost function
    cost = model.cost_function(np.array(y_test), x_test)
    print(cost)
    # Qualitative performance [Graphical]
    # display_results(x_test, y_test, y_prediction)


# @tools.debug
def load_input(input_source):
    """Load the input based on the input source."""
    x_test, y_test = utils.load_data(
        "data/{}/data_array.pkl".format(input_source))
    x_test, y_test = utils.filter_array(x_test, y_test, 0, 300)
    return x_test, y_test


# @tools.debug
def evaluate(y_test, y_prediction):
    """Computethe numerical performance."""
    # Compute confusion matrix
    result = confusion_matrix(y_test, y_prediction)
    performance = []
    # for i_index in range(len(y_test)):
    #     performance.append([y_test[i_index], y_prediction[i_index]])
    return performance


@tools.debug
def confusion_matrix(y_test, y_prediction):
    """Compute confusion matrix."""
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i_index in range(len(y_test)):
        if y_prediction[i_index] == 1: # Predict a positive
            if y_test[i_index] == 1:
                true_positive += 1
            elif y_test[i_index] == 0:
                false_positive += 1
        elif y_prediction[i_index] == 0:
            if y_test[i_index] == 1:
                false_negative += 1
            elif y_test[i_index] == 0:
                true_negative += 1
    matrix = pd.DataFrame(
        [
            [true_positive, false_positive, int(true_positive+false_positive)],
            [false_negative, true_negative, int(false_negative+true_negative)],
            [int(true_positive+false_negative), int(false_positive+true_negative), 0]
        ],
        columns=["real positive", "real negative", "total predicted"],
        index=["predict positive", "predict negative", "total real"]
    )
    precision = true_positive/(true_positive+false_positive+1)
    recall = true_positive/(true_positive+false_negative+1)
    f_score = precision*recall/(precision+recall+1)
    result = {
        "confusion_matrix": matrix,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f_score": round(f_score, 2)
    }
    return result


def display_results(x_test, y_test, y_prediction):
    """Display the results."""
    print("ROC curve to be coded......")


if __name__ == '__main__':
    test(model_string="diy")
