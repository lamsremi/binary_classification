"""
Script gathering the numerical test bench.
"""
import pandas as pd

import tools

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