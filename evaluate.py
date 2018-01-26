"""
Script for performance

- Select the model to assess the performance
"""
import pandas as pd
import numpy as np
import predict
import tools
import performance.quantitative as quantitative
import performance.qualitative as qualitative

from performance.numerical_bench import confusion_matrix

# @tools.debug
def main(model_type, input_source):
    """Perform test."""
    # Load input
    data_df = load_input(input_source)
    # Predict output
    data_df = predict_frame(
        data_df,
        model_type)
    # Quantitative performance
    performance = evaluate(data_df)


# @tools.debug
def load_input(input_source):
    """Load the input based on the input source."""
    data_df = pd.read_csv("data/{}/data.csv".format(input_source))
    return data_df

def predict_frame(data_df):
    """Predict frame."""
    predicted_data_df = data_df.copy()
    for index, serie in data_df.ietrrows():
        x_array = np.array(serie)
        predicted_data_df.loc[index, "prediction"] = predict.main(x_array)
    return data_df


@tools.debug
def evaluate(y_test, y_prediction):
    """Computethe numerical performance."""
    # Compute confusion matrix
    result = confusion_matrix(y_test, y_prediction)
    performance = []
    # for i_index in range(len(y_test)):
    #     performance.append([y_test[i_index], y_prediction[i_index]])
    return result


if __name__ == '__main__':
    MODEL_TYPE = "diy"
    INPUT_SOURCE = "us_election"
    main(MODEL_TYPE, INPUT_SOURCE)
