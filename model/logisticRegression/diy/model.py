"""
model do it yourself logistic regression
"""
import pickle
import numpy as np

import tools

class DiyLogisticReg():
    """Class DiyLogisticReg."""

    def __init__(self):
        """Init the model instance."""
        self.input_dim = 0
        self.theta_array = 0


    def predict(self, x_array):
        """
        Perform a one record prediction.
        """
        x_array = np.array(x_array)
        # If one input
        if x_array.ndim == 1:
            self.input_dim = x_array.shape[0]
            # Compute the probability
            probability = self.compute_probability(x_array)
            # Categorize it
            y_value = 1 if probability >= 0.5 else 0
            # Return the value
            return y_value
        # If an array of inputs
        elif x_array.ndim == 2:
            y_array = []
            self.input_dim = x_array.shape[1]
            for x_vect in x_array:
                # Compute the probability
                probability = self.compute_probability(x_vect)
                # Categorize it
                y_value = 1 if probability >= 0.5 else 0
                y_array.append(y_value)
            y_array = np.array(y_array)
            return y_array

    # ----- SIGMOID FUNCTION ---------------------------------------------------------

    # @tools.debug
    def compute_probability(self, x_vect):
        """Compute probability."""
        # print("--------------------------> x_vect : {}".format(x_vect))
        probability = 1 / (1 + np.exp(-1*np.dot(
            x_vect.reshape([1, self.input_dim]),
            self.theta_array.reshape([self.input_dim, 1])
        )))
        return probability

    # ---------------------------------------------------------------------------------

    # @tools.debug
    def fit(
            self,
            x_array,
            y_array,
            alpha=0.0005,
            n_iter=100
        ):
        """
        Train the model.
        """
        self.input_dim = x_array.shape[1]
        self.theta_array = np.zeros(self.input_dim)

        x_array = np.array(x_array)
        y_array = np.array(y_array)
        # print(x_array.shape)
        alpha = 0.0005
        x_arrays = self.create_batches(x_array, batch_size=50)
        y_arrays = self.create_batches(y_array, batch_size=50)
        for i_iter in range(n_iter):
            theta_arrays = []
            for x_arr, y_arr in zip(x_arrays, y_arrays):
                theta_arrays.append(self.update_weights(alpha, x_arr, y_arr))
            self.theta_array = np.mean(theta_arrays, axis=0)
            print("Cost at iteration {} : {}".format(i_iter, self.cost_function(x_array, y_array)))

    @staticmethod
    # @tools.debug
    def create_batches(array, batch_size):
        """Create batches."""
        n_records = array.shape[0]
        arrays = []
        for index in range(n_records//batch_size):
            arrays.append(array[index*50:(index+1)*50])
        arrays.append(array[n_records//batch_size:])
        return arrays

    # @tools.debug
    def update_weights(self, alpha, x_array, y_array):
        """
        Update the weights.
        """
        weights = np.add(
            self.theta_array.reshape([1, self.input_dim]),
            -alpha*self.derivative_cost_function(x_array, y_array).reshape([1, self.input_dim])
        )
        return weights[0]

    # @tools.debug
    def cost_function(self, x_array, y_array):
        """Compute the cost function."""
        m_value = len(y_array)
        y_probabilities = np.array([self.compute_probability(x_vect) for x_vect in x_array])
        # print(y_probabilities)
        cost = -1/m_value*np.add(
            np.dot(
                y_array.reshape([1, m_value]),
                np.log(y_probabilities).reshape([m_value, 1])
            ).reshape([1, 1]),
            np.dot(
                1-y_array.reshape([1, m_value]),
                np.log(1-y_probabilities).reshape([m_value, 1])
            ).reshape([1, 1])
        )[0]
        return cost

    # @tools.debug
    def derivative_cost_function(self, x_array, y_array):
        """
        The derivative of the cost function with respect
        to each of the theta element.
        """
        m_value = len(y_array)

        derivative_vector = np.array(
            [
                -1/m_value*(
                    np.add(
                        np.dot(
                            y_array.reshape([1, m_value]),
                            np.multiply(
                                x_array[:, j_index].reshape([x_array.shape[0], 1]),
                                1 + np.exp(np.dot(
                                    x_array.reshape([x_array.shape[0], self.input_dim]),
                                    self.theta_array.reshape([self.input_dim, 1])
                                ))
                            ).reshape([m_value, 1])
                        ),
                        np.dot(
                            (1 - y_array).reshape([1, m_value]),
                            np.multiply(
                                - x_array[:, j_index].reshape([x_array.shape[0], 1]),
                                1 + np.exp(-np.dot(
                                    x_array.reshape([x_array.shape[0], self.input_dim]),
                                    self.theta_array.reshape([self.input_dim, 1])
                                ))
                            ).reshape([m_value, 1])
                        ),
                    )
                )[0]
                for j_index in list(range(self.input_dim))
            ]
        )
        return derivative_vector

    # @tools.debug
    def persist(self, path_pickle):
        """Save the model."""
        pickle.dump(self.theta_array, open(path_pickle, "wb"))

    def load(self, path_pickle):
        """Load the model."""
        self.theta_array = pickle.load(open(path_pickle, 'rb'))
