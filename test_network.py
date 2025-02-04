import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
## new:
from sklearn.metrics import confusion_matrix

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


# Testing the network
#### Modify the code to get the confusion matrix ####
all_preds = []
true_labels = [] 

# hint: 
#     you can use confusion_matrix from sklearn.metrics (pip install -U scikit-learn)
#     to compute the confusion matrix. Or you can write your own code :)

# I used these resources to code this part
# https://www.w3schools.com/python/python_ml_confusion_matrix.asp
# https://www.jcchouinard.com/confusion-matrix-in-scikit-learn/
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html


for i in range(0, xtest.shape[1], 100):
    _, P = convnet_forward(params, layers, xtest[:, i:i + 100], test=True)
    preds = np.argmax(P, axis=0)
    all_preds.extend(preds)
    true_labels.extend(ytest[:, i:i + 100].ravel())

# Compute the confusion matrix
cm = confusion_matrix(true_labels, all_preds)

# Print the confusion matrix to the terminal
np.set_printoptions(precision=2)  
print("Confusion Matrix:")
print(cm)

# the most confused pairs are (5,6) and (7,2)
