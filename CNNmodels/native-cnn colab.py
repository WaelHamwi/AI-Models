# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 00:25:15 2022

@author: WAEL
"""

#!pip install split-folders
#!pip install tensorflow-gpu
from tensorflow.keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop 
from keras.layers import Input, Lambda, Dense, Flatten,Conv2D,MaxPool2D,Dropout,Activation
from keras.models import Model
import numpy as np  
from sklearn.model_selection import train_test_split 
#import splitfolders
from tensorflow.keras.optimizers import RMSprop 
#database = (r'/content/drive/MyDrive/X-ray (1)')
#databaseOut=(r'/content/drive/MyDrive/X-ray (2)')
#splitfolders.ratio(database,databaseOut,seed=1337,ratio=(0.8,0.2))
train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)
train_dataset=train.flow_from_directory(r'/content/drive/MyDrive/X-ray (2)/train',
                                        target_size=(224,224),
                                        batch_size=3,
                                        class_mode='binary'
    )
validation_dataset=validation.flow_from_directory(r'/content/drive/MyDrive/X-ray (2)/val',
                                        target_size=(224,224),
                                        batch_size=3,
                                        class_mode='binary'
    )

print(train_dataset.class_indices)
print(validation_dataset.class_indices)
print(train_dataset.classes)

dense_layers = [0]
layer_sizes = [32]
conv_layers = [2]



model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=32,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(input_shape=(224,224,3),filters=32,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten()) # flatten will take the input as it is

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy'])
model_fit=model.fit(train_dataset,epochs=10,validation_data=validation_dataset)