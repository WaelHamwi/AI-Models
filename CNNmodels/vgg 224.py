# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 05:45:56 2022

@author: WAEL
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import random
import pickle

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from keras.applications.vgg16 import VGG16
import numpy as np
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.callbacks import TensorBoard

train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)
train_dataset=train.flow_from_directory(r'C:\Users\WAEL\Desktop\train\databaseOut\train',
                                        target_size=(224,224),
                                        batch_size=3,
                                        class_mode='binary'
    )
validation_dataset=train.flow_from_directory(r'C:\Users\WAEL\Desktop\train\databaseOut\val',
                                        target_size=(224,224),
                                        batch_size=3,
                                        class_mode='binary'
    )
model = Sequential()

model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=10, activation="softmax")) 
img1_shape = (224,224,3)
model = model(include_top=False, weights='imagenet', input_shape=img1_shape)
model.compile(loss='binary_crossentropy',
              optimizer= RMSprop(lr=0.001),
              metrics=['accuracy'])
model_fit=model.fit(train_dataset,epochs=10,validation_data=validation_dataset)