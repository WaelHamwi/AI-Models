# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 17:42:42 2022

@author: WAEL
"""
from keras.layers import Input, Lambda, Dense, Flatten
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pickle
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten,Dropout
train_path = r'C:\Users\WAEL\Desktop\train\databaseOut\train'
valid_path = r'C:\Users\WAEL\Desktop\train\databaseOut\val'

model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=32,kernel_size=(3,3), activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))    

model.add(Conv2D(filters=64,kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128,kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=64,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation="softmax"))
model.summary()

model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
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
train_dataset.class_indices
model_fit=model.fit(train_dataset,epochs=10,validation_data=validation_dataset)