"""
Created on Thu Jan  6 20:37:33 2022

@author: WAEL
"""
import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Lambda, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)
train_dataset=train.flow_from_directory(r'/content/drive/MyDrive/X-ray (2)/train',
                                        target_size=(224,224),
                                        batch_size=3,
                                        class_mode='binary'
                                        
    )
validation_dataset=train.flow_from_directory(r'/content/drive/MyDrive/X-ray (2)/val',
                                        target_size=(224,224),
                                        batch_size=3,
                                        class_mode='binary'
                                        
    )

#normalizing the data

model = tf.keras.applications.mobilenet.MobileNet()

base_input = model.layers[0].input
base_output = model.layers[-4].output
#adding other layers
flat_layer = layers.Flatten()(base_output)
final_output =layers.Dense(1)(flat_layer) #output is either 1 or 0
final_output= layers.Activation('sigmoid')(final_output)

base_model= tf.keras.applications.mobilenet.MobileNet( include_top=False, input_shape=(224,224,3), pooling='max', weights='imagenet',dropout=.4) 
x=base_model.output
final_output=layers.Dense(1, activation='sigmoid')(x)
new_model = Model(inputs =base_model.input, outputs = final_output)
new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
new_model.fit(train_dataset,epochs=10,validation_data=validation_dataset)
