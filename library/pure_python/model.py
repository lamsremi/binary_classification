"""Implementation of logistic regression in pure python.
"""
import os
import pickle

import tools
from library.pure_python.utils import cost, cost_derivative, product, sigmoid


class Model():
    """Class DIYLogisticReg."""

    def __init__(self):
        """Init the model instance.
        Attributes:
            _input_dim
            _theta_array
            _biais
        """
        self._input_dim = None
        self._theta_array = None
        self._biais = None


    def predict(self, inputs_data):
        """Perfom a prediction.
        Args:
            inputs_data (list): list of records to predict.
        Return:
            prediction_df (list):
        """
        # Return the prediction list
        return [self._predict_instance(row) for row in inputs_data]


    def _predict_instance(self, x_array):
        """Predict one record.
        Args:
            row (list): input of one record.
        Return:
            prediction (float): prediction.
        """
        # Compute the produict
        my_product = product(x_array,
                             self._theta_array,
                             self._biais)
        # Compute the probability
        probability = sigmoid(my_product)
        # Categorize it
        y_value = 1 if probability >= 0.5 else 0
        # Return the value
        return y_value


    def load_parameters(self, model_version):
        """Load the model's parameters.
        """
        # If a model version is given
        if model_version:

            # Set the folder path
            folder_path = "library/pure_python/params/" + model_version
            # Load the input dimension
            with open(folder_path + "/input_dim.pkl", "rb") as handle:
                self._input_dim = pickle.load(handle)
            # Load the weights
            with open(folder_path + "/weights.pkl", "rb") as handle:
                self._theta_array = pickle.load(handle)
            # Load the biais
            with open(folder_path + "/biais.pkl", "rb") as handle:
                self._biais = pickle.load(handle)


    def fit(self,
            labeled_data,
            alpha,
            epochs):
        """Fit the parameters of the model.
        Args:
            labeled_data (list): labeled dataset for fitting.
            [
                ([ 0.0, 6.0, 6.0, 2.0], 1.0)
                ([ 8.0, 7.0, 4.0, 2.0], 1.0)
            ]
        """
        # Extract input and labels
        x_arrays = [instance[0] for instance in labeled_data]
        y_arrays = [instance[1] for instance in labeled_data]

        # If starting from nothing
        if self._input_dim is None:

            # Set input dimension
            self._input_dim = len(x_arrays[0])
            # Initializa theta
            self._theta_array = [0 for itn in range(self._input_dim)]
            # Initialize biais
            self._biais = 0

        # Loop through iteration
        for epoch in range(epochs):

            # Compute the cost
            my_cost = cost(x_arrays,y_arrays,
                        self._theta_array, self._biais)
            # Display the cost
            print("epoch {} - cost : {}".format(epoch, my_cost))
            # Compute derivative weights
            weights_derivative, biais_derivative = cost_derivative(
                x_arrays,
                y_arrays,
                self._theta_array,
                self._biais)
            # Update weights parameters
            self._theta_array = [
                theta + weight_der for theta, weight_der
                in zip(self._theta_array, [alpha*x for x in weights_derivative])
            ]
            # Update biais
            self._biais += alpha * biais_derivative


    # @tools.debug
    def persist_parameters(self, model_version):
        """Persist model's parameters.
        Args:
            model_version (str): version of model to store
        """
        # Set the folder path
        folder_path = "library/pure_python/params/" + model_version

        # Check if folder exists
        if not os.path.exists(folder_path):
            # Create the folder
            os.mkdir(folder_path)

        # Persist the input dimension
        with open(folder_path + "/input_dim.pkl", "wb") as handle:
            pickle.dump(self._input_dim, handle)
        # Persist the weights
        with open(folder_path + "/weights.pkl", "wb") as handle:
            pickle.dump(self._theta_array, handle)
        # Persist the biais
        with open(folder_path + "/biais.pkl", "wb") as handle:
            pickle.dump(self._biais, handle)
