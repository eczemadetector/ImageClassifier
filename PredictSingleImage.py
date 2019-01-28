import numpy as np
import sys
import os
import cv2
from PIL import Image

# Hide useless warnings
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from keras.models import Model,load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Hide useless warnings
sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#define the input file
file = sys.argv[1]
#load trained model
model = load_model('ResNet50.h5')

input_image = load_img(file)  # this is a PIL image

# Convert to Numpy Array
input_image = img_to_array(input_image)
input_image = input_image.reshape(1,128,128,3)

# Run prediction algorithm and print results
pred = model.predict(input_image)
print(int(pred[0]))
