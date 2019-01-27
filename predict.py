from keras.models import load_model
import cv2
import numpy as np

#read image and reshape
img = cv2.imread('InputImage.png')
img = cv2.resize(img,(128,128))
img = np.expand_dims(img,axis=0)

#load model and predict
model = load_model('ResNet50.h5')
classes = model.predict(img)

print (classes)
