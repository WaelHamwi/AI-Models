# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:41:12 2022

@author: WAEL
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
from keras.utils.vis_utils import plot_model
#from keras.utils.vis_utils_util import plot_model
import visualkeras
from eiffel2 import builder
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array , load_img
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from numpy import expand_dims
import cv2
model= tf.keras.models.load_model("32x2x0CNN.model")
def prepare(filepath):
    IMG_SIZE = 500
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
   
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array / 255.0
    print(new_array)
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    
    






# summarize filter shapes
for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
# get filter weights
filters, biases = layer.get_weights()
print(layer.name, filters.shape)



# summarize feature map shapes
for i in range(len(model.layers)):
	layer = model.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)       
model = Model(inputs=model.inputs, outputs=layer)
# load the image with the required shape
img = load_img('2.png', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = prepare(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot the output from each block
square = 8
for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
			ix += 1
	# show the figure
	pyplot.show()    
  