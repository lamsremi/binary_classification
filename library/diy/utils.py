"""Module including the utilities functions:
* probability function
* cost function
"""
import numpy as np

import tools


def cost_function(x_arrays,
                  y_arrays,
                  weights,
                  biais):
    """Compute the cost function.
    Args:
        x_arrays (ndarray): list of inputs
            shape [number_inputs, input_dimension]
        y_arrays (ndarray):
            shape [number_inputs, 1]
        weights (ndarray):
        biais (float):
    Return:
        cost (float): overall cost for the given batch.
    Note:
        * Here use cross entropy instead of usual mean square error
    """
    # Initalize cost
    cost = 0

    # For each of the array of newly shape [input_dimension, ] and [1, ]
    for x_array, y_array in zip(x_arrays, y_arrays):

        # Reshape input from [input_dimension, ] to [1, input_dim]
        x_array = x_array.reshape(1, -1)

        # Reshape output from [1, ] to None
        y_value = y_array[0]

        # Compute product
        product = compute_product(x_array, weights, biais)

        # Compute sigmoid
        sigmoid = compute_sigmoid(x_value=product)

        # Compute the cost for this record
        record_cost = cross_entropy(x_value=sigmoid, constant=y_value)

        # Add to the cost
        cost += record_cost

    # Divide by the total number of inputs
    cost /= float(-x_arrays.shape[0])

    # Return cost
    return cost

# @tools.debug
def compute_product(x_array, weights, biais):
    """Compute probability.
    Args:
        x_array (ndarray): input array shape (1, input_dim)
        weights (nparray): weights of the model shape (1, input_dim)
        biais (float): biais parameter.
    Return:
        probability (float): computed probability.
    """
    # Compute the product
    result = np.dot(x_array, weights.transpose())[0][0]
    # Add the biais
    result = result + biais
    # Return result
    return result

def derive_weights_product(x_array, weights, biais):
    """Compute derivative with respect to each of the weights.
    """
    # Set
    derivative_product = x_array.copy()
    # Return
    return derivative_product

def derive_biais_product(x_array, weights, biais):
    """Compute derivative with respect to each of the weights.
    """
    # Set
    derivative_product = 1
    # Return
    return derivative_product

def compute_sigmoid(x_value):
    """Compute sigmoid.
    """
    # Compute the sigmoid
    y_value = 1.0 / (1 + np.exp(-x_value))
    # Return the result
    return y_value

def derive_compute_sigmoid(x_value):
    """Compute sigmoid.
    """
    # Compute the sigmoid
    derivative = -np.exp(-x_value) / (1 + np.exp(-x_value))**2
    # Return the result
    return derivative

def cross_entropy(x_value, constant):
    """Cross entropy function.
    Args:
        x_value (float)
        constant (float)
    Return:
        y_value (float)
    """
    # Compute
    y_value = constant*np.log(x_value) + (1-constant)*np.log(1-x_value)
    # Return
    return y_value

def derive_cross_entropy(x_value, constant):
    """Derivative cross entropy.
    Return:
        derivative (float)
    """
    # Compute derivative
    derivative = constant/x_value + (1-constant)*(-1)/(1-x_value)
    # Return value
    return derivative


def derivative_cost_function(x_arrays,
                             y_arrays,
                             weights,
                             biais):
    """Compute the derivative of the cost function
    with respect to each of the weights.
    Args:
        x_arrays (ndarray): list of inputs
                            shape [number_inputs, input_dimension]
        y_arrays (ndarray): list of output
                            shape [number_inputs, 1]
        weights (ndarray):
        biais (float):
    Return:
        derivative_weights (ndarray): array of derivatives with respect to
                                      each of the weights of
                                      shape [1, input_dimension]
        derivative_biais (float): derivative with respect to the biais
    """
    # Initialize the vector derivatives of weights
    derivative_weights = np.zeros([1, x_arrays.shape[1]])

    # Initialize the derivative of biais
    derivative_biais = 0

    # For each of the array of newly shape [input_dimension, ] and [1, ]
    for x_array, y_array in zip(x_arrays, y_arrays):

        # Reshape input from [input_dimension, ] to [1, input_dim]
        x_array = x_array.reshape(1, -1)

        # Reshape output from [1, ] to None
        y_value = y_array[0]

        # Comput the derivative with respect to the weights
        derivative_weights_product = derive_weights_product(x_array,
                                                               weights,
                                                               biais)

        # Compute the derivative with respect to the biais
        derivative_biais_product = derive_biais_product(x_array,
                                                            weights,
                                                            biais)

        # Compute product for derivative sigmoid
        product = compute_product(x_array, weights, biais)

        # Derivative sigmoid
        derivative_sigmoid = derive_compute_sigmoid(
            x_value=product)

        # Compute product for sigmoid
        product = compute_product(x_array, weights, biais)

        # Compute sigmoid for derivative cross entropy
        sigmoid = compute_sigmoid(x_value=product)

        # Derivative cross entropy
        derivative_cross = derive_cross_entropy(
            x_value=sigmoid, constant=y_value)

        # Derivative record weights
        record_weights_derivative = derivative_weights_product*derivative_sigmoid*derivative_cross

        # Derivative record weights
        record_biais_derivative = derivative_biais_product*derivative_sigmoid*derivative_cross

        # Add on to the weights one
        derivative_weights += record_weights_derivative

        # Add on to the biais one
        derivative_biais += record_biais_derivative

    # Divide by the total number of inputs for weights
    derivative_weights /= float(-x_arrays.shape[0])

    # Same for biais
    derivative_biais /= float(-x_arrays.shape[0])

    return derivative_weights, derivative_biais




