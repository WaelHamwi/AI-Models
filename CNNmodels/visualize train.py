# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 19:44:43 2022
@author: WAEL
"""
#from keras.utils.vis_utils_util import plot_model
from matplotlib import pyplot as plt
from keras.models import Model
import tensorflow as tf
from test_data import prepare
from train2 import model

layer = model.layers #Conv layers at 0, 3
print(layer)
filters, biases = model.layers[3].get_weights()
print(layer[1].name, filters.shape)            
fig1=plt.figure(figsize=(8, 12))
columns = 8
rows = 4
n_filters = columns * rows
for i in range(1, n_filters +1):
    f = filters[:, :, :, i-1]
    fig1 =plt.subplot(rows, columns, i)
    fig1.set_xticks([])  #Turn off axis
    fig1.set_yticks([])
    plt.imshow(f[:, :, 0], cmap='gray') #Show only the filters from 0th channel (R)
    #ix += 1
plt.show()    
conv_layer_index = [0, 3]
outputs = [model.layers[i].output for i in conv_layer_index]
model_short = Model(inputs=model.inputs, outputs=outputs)
print(model_short.summary())

model = tf.keras.models.load_model("32x2x0CNN.model")

feature_output = model_short.predict([prepare('2.png')]) #2 convolutional layers
#num of filters is 32
#columns = 8
#rows = 4
for ftr in feature_output:
    #pos = 1
    fig=plt.figure(figsize=(12, 12))
    for i in range(1, columns*rows +1):
        fig =plt.subplot(rows, columns, i)
        fig.set_xticks([])  #Turn off axis
        fig.set_yticks([])
        plt.imshow(ftr[0, :, :, i-1], cmap='gray')
        #pos += 1
    plt.show()