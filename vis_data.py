import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt

# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)
m_train = xtrain.shape[1]

batch_size = 1
layers[0]['batch_size'] = batch_size

img = xtest[:,0]
img = np.reshape(img, (28, 28), order='F')
plt.imshow(img.T, cmap='gray')
plt.show()

# output = convnet_forward(params, layers, xtest[:,0:1])
# output_1 = np.reshape(output[0]['data'], (28,28), order='F')

##### Fill in your code here to plot the features ######
def visualize_layer_output(output, layer_idx, title):
    layer_output_data = output[layer_idx]['data']
    
    # Get the layer configuration from utils.py
    layer_config = layers[layer_idx]
    
    # If the current layer is relu, get the configuration from conv
    if layer_config['type'] == 'RELU':
        layer_config = layers[layer_idx - 1]
        num_filters = layer_config.get('num', 20)
        h_out = ((layers[0]['height'] - layer_config['k']) // layer_config['stride']) + 1
        w_out = ((layers[0]['width'] - layer_config['k']) // layer_config['stride']) + 1
    
    else:  # For conv layer
        num_filters = layer_config.get('num', 20)
        h_out = ((layers[0]['height'] - layer_config['k']) // layer_config['stride']) + 1
        w_out = ((layers[0]['width'] - layer_config['k']) // layer_config['stride']) + 1
    
    # Reshape layer_output_data to [h_out, w_out, num_filters]
    layer_output_data = layer_output_data.reshape((h_out, w_out, num_filters), order='F')
    
    plt.figure(figsize=(10,8))
    plt.suptitle(title)
    
    for i in range(min(20, num_filters)):
        plt.subplot(4, 5, i + 1)
        plt.imshow(np.transpose(layer_output_data[:, :, i], (1, 0)), cmap='gray')
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

# Running forward pass to get output from layers
output = convnet_forward(params, layers, xtest[:,0:1])

# Visualize conv layer
visualize_layer_output(output, 1, 'Features after CONV Layer')

# Visualize relu layer
visualize_layer_output(output, 2, 'Features after ReLU Layer')