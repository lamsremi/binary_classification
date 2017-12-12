"""
model do it yourself logistic regression
"""
import numpy as np

class DiyLogisticReg():
    """Class DiyLogisticReg."""

    def __init__(self):
        """Init the model instance."""
        self.theta_array = [-1, -1, 0, -1, 0, 1, -1, 1]

    def predict(self, x_array):
        """
        Perform a one record prediction.
        """
        x_array = np.array(x_array)
        # If one input
        if x_array.ndim == 1:
            # Compute the probability
            probability = self.compute_probability(x_array)
            # Categorize it
            y_value = 1 if probability >= 0.5 else 0
            # Return the value
            return y_value
        # If a list of inputs
        elif x_array.ndim == 2:
            y_array = []
            for x_vect in x_array:
                # Compute the probability
                probability = self.compute_probability(x_vect)
                # Categorize it
                y_value = 1 if probability >= 0.5 else 0
                y_array.append(y_value)
            y_array = np.array(y_array)
            return y_array

    def compute_probability(self, x_vect):
        """Compute probability."""
        probability = 1 / (1 + np.exp(-1*np.dot(x_vect, self.theta_array)))
        return probability

    def fit(self):
        """
        Train the model.
        """
        print("Training of the diy model to be coded")

    def cost_function(self, y_array, x_array):
        """Compute the cost function."""
        m_value = len(y_array)
        y_probabilities = np.array([self.compute_probability(x_vect) for x_vect in x_array])
        sum_1 = np.dot(y_array, np.log10(y_probabilities))
        sum_2 = np.dot(1-y_array, np.log10(1-y_probabilities))
        cost = -1/m_value*(sum_1 + sum_2)
        return cost

    def persist(self, path_pickle):
        """Save the model."""
        print("Persistence of the diy model to be coded")

