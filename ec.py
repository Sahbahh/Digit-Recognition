import cv2
import numpy as np
from scipy.io import loadmat
from conv_net import convnet_forward
from utils import get_lenet
from init_convnet import init_convnet


# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

# Map the raw parameters to the initialized network parameters
for params_idx in range(len(params)):
    raw_w = params_raw[0, params_idx][0, 0][0]
    raw_b = params_raw[0, params_idx][0, 0][1]
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Assume that labels are the digits 0 to 9
labels = list(range(10))

# list of images to process
image_paths = ['../images/image1.JPG', '../images/image2.JPG', '../images/image3.png', '../images/image4.JPG'] 

for image_path in image_paths:
    predicted_values = []
    
    print(f"Loading {image_path}")
    # Read the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform thresholding
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Display Thresholded Image
    # cv2.imshow('Thresholded Image', thresh)
    # cv2.waitKey(0)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left-to-right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    area_threshold_fraction = 0.0023  ## to work for image4
    image_area = image.shape[0] * image.shape[1]
    area_threshold = area_threshold_fraction * image_area

    # print(f"Processing {image_path}")  # To check which image it is processing
    # print(f"Area Threshold: {area_threshold}")  # To check the calculated area threshold value

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # print(f"Contour Area: {w * h}")
        
        # Skip small noise regions
        if w * h < area_threshold:  
            continue
        # print("Drawing bounding box")

        # Draw bounding box around each digit on the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0,255), 2) 

        # Extract the digit
        digit = thresh[y:y+h, x:x+w]
        
        # Pad the digit to make it square
        size = max(w, h)
        padding = np.zeros((size, size), dtype=np.uint8)
        x_offset, y_offset = (size - w) // 2, (size - h) // 2
        padding[y_offset:y_offset+h, x_offset:x_offset+w] = digit
        
        # Resize the digit to 28x28 and normalize the pixel values
        digit_resized = cv2.resize(padding, (28, 28))
        digit_normalized = digit_resized / 255.0
        
        # Ensure batch size is 1 in layers configuration for each digit processed
        layers[0]['batch_size'] = 1
        
        # Reshape to match the input shape of the network
        digit_final = digit_normalized.reshape(1, 28 * 28).T
        
        # Pass the processed digit through the network
        _, P = convnet_forward(params, layers, digit_final, test=True)  # Ensure to get the Probability vector P
        
        # Find the label with maximum probability
        predicted_label = labels[np.argmax(P[:, 0], axis=0)]  # Assuming the probability vector is column-wise
        predicted_values.append(predicted_label)

    # printing the image with bound boxes
    cv2.imshow('Bounding Boxes', image)
    cv2.waitKey(0)    
    cv2.destroyAllWindows()  # Close all OpenCVs

    # Print the predicted values for each image
    print(f"Predicted Values for {image_path.split('/')[-1]}:")
    print(predicted_values)
