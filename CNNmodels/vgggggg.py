# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 18:53:29 2022

@author: WAEL
"""

from tensorflow.keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop 
from keras.layers import Input, Lambda, Dense, Flatten,Conv2D,MaxPool2D
from keras.models import Model
img=image.load_img('2.png')
plt.imshow(img)
#print(cv2.imread('2.png'))
print(cv2.imread('2.png').shape)

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
#train_dataset=np.asarray(train_labels).astype('float32').reshape((-1, 1))
#validdation_dataset=np.asarray(train_labels).astype('float32').reshape((-1, 1))

print(train_dataset.class_indices)
print(validation_dataset.class_indices)
print(train_dataset.classes)

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
model.add(Dense(units=1, activation="softmax"))
#opt = Adam(lr=0.001)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model_fit=model.fit(train_dataset,epochs=10,validation_data=validation_dataset)

