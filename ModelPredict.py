from ModelClasses import *
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from ModelCode import fashion_mnist_labels

labels = os.listdir('PokemonData')

# Read an image
# image_data = cv2.imread('TestImages/test.jpg', cv2.IMREAD_GRAYSCALE)
image_data = load_img('TestImages/test.jpg', target_size=(128,128))
# Resize to the same size as Fashion MNIST images
# image_data = cv2.resize(image_data, (128, 128))

# Invert image colors
# image_data = 255 - image_data
image_data=img_to_array(image_data)
img_data=image_data/255.0
# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Load the model
model = Model.load('pokemon.model')

# Predict on the image
confidences = model.predict(image_data)

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
prediction = labels[predictions[0]]

print(prediction)