import numpy as np
from utils import get_lenet
from init_convnet import init_convnet
from conv_net import convnet_forward
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2

# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Initialize the lists for predicted and truth values
predicted_values = []
truth_values = list(range(4,9))  # testing 4.jpg to 9.jpg

# Loop over each image
for i in range(4,9):
    image_path = f'../data/{i}.jpg'
    
    # Read the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # # Verify if the image is read and converted correctly
    # print(f"Image {i} Shape: {image.shape} Sum: {np.sum(image)}")
    
    # Normalizing 
    image = image / 255.0  
    image_array = np.expand_dims(image, axis=-1)
    
    dummy_batch_size = 100
    image_array = np.tile(image_array, (1, 1, 1, dummy_batch_size))
    
    _, P = convnet_forward(params, layers, image_array, test=True)
    predicted_label = np.argmax(P[:, 0], axis=0)
    predicted_values.append(predicted_label)

print("Predicted Values:")
print(predicted_values)
print("Truth Values:")
print(truth_values)
