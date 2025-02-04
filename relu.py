import numpy as np

def relu_forward(input_data):
    output = {
        'height': input_data['height'],
        'width': input_data['width'],
        'channel': input_data['channel'],
        'batch_size': input_data['batch_size'],
    }

    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    output['data'] = np.maximum(input_data['data'],0)
    
    return output

def relu_backward(output, input_data, layer):
    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    input_od = np.zeros_like(input_data['data'])

    input_od = np.zeros_like(input_data['data'])
    active_positions = input_data['data'] >= 0  ## check
    input_od[active_positions] = output['diff'][active_positions]

    return input_od
