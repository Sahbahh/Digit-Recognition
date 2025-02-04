import numpy as np


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """

    d, k = input["data"].shape
    n = param["w"].shape[1]

    ###### Fill in the code here ######
    w = param["w"]
    b = param["b"]

    # f(x)=wx+b

    ##### COME BACK AND CHECK!!!!!
    # using np.newaxis to convert 1D array into a 2D column vector
    # outputData = np.dot(w, input["data"]) + b[:, np.newaxis]
    outputData = np.dot(param["w"].T, input["data"]) + param["b"].T

    # Initialize output data structure
    output = {
        "height": n,
        "width": 1,
        "channel": 1,
        "batch_size": k,
        "data":outputData # replace 'data' value with your implementation
    }

    return output


def inner_product_backward(output, input_data, layer, param):
    """
    Backward pass of inner product layer.

    Parameters:
    - output (dict): Contains the output data.
    - input_data (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """

    # Debugging to find dimensions
    # print("Shape of param['w']:", param['w'].shape)
    # print("Shape of output['diff']:", output['diff'].shape)

    param_grad = {}
    ###### Fill in the code here ######
    # Replace the following lines with your implementation.
    param_grad['w'] = np.dot(input_data['data'], output['diff'].T)  # shape: (800, 500)
    
    # Gradient with respect to biases
    param_grad['b'] = np.sum(output['diff'], axis=1)
    
    # Gradient with respect to the input
    input_od = np.dot(param['w'], output['diff'])  # shape: (800, 100)

    return param_grad, input_od