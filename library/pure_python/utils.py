"""Utilities functions used for training and predicting:
* probability function
* cost function
"""
import math

import tools


def cost(x_arrays,
         y_arrays,
         weights,
         biais):
    """Compute the cost function.
    Args:
        x_arrays (list): list of inputs shape [number_inputs, input_dimension]
        y_arrays (list): list of output shape [number_inputs]
        weights (list): weights.
        biais (float): biais.
    Return:
        cost (float): overall cost for the given batch.
    Note:
        * Here use cross entropy instead of usual mean square error.
    """
    # Initalize cost
    cost = 0

    # For each of the array of newly shape [input_dimension, ] and [1, ]
    for x_array, y_value in zip(x_arrays, y_arrays):
        # Compute sigmoid of the product
        my_sigmoid = sigmoid(product(x_array, weights, biais))
        # Compute the cost for this record
        record_cost = cross_entropy(my_sigmoid, y_value)
        # Add to the cost
        cost += record_cost

    # Divide by the total number of inputs
    cost /= float(-len(x_arrays))
    # Return cost
    return cost


# @tools.debug
def product(x_array, weights, biais):
    """Compute the matrix product between x_array and weights and add the biais."""
    return sum([x*w for x,w in zip(x_array, weights)]) + biais


def sigmoid(x_value):
    """Compute sigmoid."""
    return  1.0 / (1 + math.exp(-x_value))


def cross_entropy(x_value, constant):
    """Compute the cross entropy."""
    return constant*math.log(x_value) + (1-constant)*math.log(1-x_value)


def derive_weights_product(x_array, weights, biais):
    """Compute derivative with respect to each of the weights."""
    return x_array


def derive_biais_product(x_array, weights, biais):
    """Compute derivative with respect to the biais."""
    return 1


def derive_sigmoid(x_value):
    """Compute sigmoid derivative."""
    return -math.exp(-x_value) / (1 + math.exp(-x_value))**2


def derive_cross_entropy(x_value, constant):
    """Derivative cross entropy."""
    return constant/x_value + (1-constant)*(-1)/(1-x_value)


def cost_derivative(x_arrays, y_arrays,
                    weights, biais):
    """Compute the derivative of the cost function
    with respect to each of the weights.
    Args:
        x_arrays (list): list of inputs
        y_arrays (ndarray): list of outputs
        weights (list): weights.
        biais (float): biais.
    Return:
        derivative_weights (list): list of derivatives with respect to
                                      each of the weights of
                                      shape [1, input_dimension]
        derivative_biais (float): derivative with respect to the biais
    """
    # Initialize the vector derivatives of weights and of biais
    derivative_weights = [0 for itn in range(len(x_arrays[0]))]
    derivative_biais = 0

    # For each of the array of newly shape [input_dimension, ] and [1, ]
    for x_array, y_value in zip(x_arrays, y_arrays):
        # Comput the derivative with respect to the weights
        derivative_weights_product = derive_weights_product(x_array, weights, biais)
        # Compute the derivative with respect to the biais
        derivative_biais_product = derive_biais_product(x_array, weights, biais)
        # Compute product for derivative sigmoid
        my_product = product(x_array, weights, biais)
        # Derivative sigmoid
        derivative_sigmoid = derive_sigmoid(my_product)
        # Compute sigmoid for derivative cross entropy
        my_sigmoid = sigmoid(my_product)
        # Derivative cross entropy
        derivative_cross = derive_cross_entropy(my_sigmoid, y_value)
        # Derivative record weights
        record_weights_derivative = [x*derivative_sigmoid*derivative_cross for x in derivative_weights_product]
        # Derivative record weights
        record_biais_derivative = derivative_biais_product*derivative_sigmoid*derivative_cross
        # Add on to the weights one
        derivative_weights = [
            x+y for x,y in zip(derivative_weights, record_weights_derivative)
        ]
        # Add on to the biais one
        derivative_biais += record_biais_derivative

    # Divide by the total number of inputs for weights and for biais
    derivative_weights = [x/float(-len(x_arrays)) for x in derivative_weights]
    derivative_biais /= float(-len(x_arrays))

    return derivative_weights, derivative_biais
