# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:34:34 2022

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
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
img=image.load_img('2.png')
plt.imshow(img)
#print(cv2.imread('2.png'))
print(cv2.imread('2.png').shape)
train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)
train_dataset=train.flow_from_directory(r'C:\Users\WAEL\Desktop\train\databaseOut\train',
                                        target_size=(200,200),
                                        batch_size=3,
                                        class_mode='binary'
    )
validation_dataset=validation.flow_from_directory(r'C:\Users\WAEL\Desktop\train\databaseOut\val',
                                        target_size=(200,200),
                                        batch_size=3,
                                        class_mode='binary'
    )
print(train_dataset.class_indices)
print(validation_dataset.class_indices)
print(train_dataset.classes)

model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(input_shape=(200,200,3),filters=16,kernel_size=(3,3), activation="relu"),
                                  tf.keras.layers.MaxPool2D(2,2),#
                                  tf.keras.layers.Conv2D(32,(3,3),activation= 'relu'),tf.keras.layers.MaxPool2D(2,2),#
                                  tf.keras.layers.Conv2D(64,(3,3),activation= 'relu'),#
                                  tf.keras.layers.MaxPool2D(2,2),#
                                  tf.keras.layers.Flatten(),#
                                  tf.keras.layers.Dense(512,activation= 'relu'),#
                                  tf.keras.layers.Dense(1,activation= 'sigmoid')
    ])
model.compile(loss='binary_crossentropy',
              optimizer= RMSprop(lr=0.001),
              metrics=['accuracy'])
model_fit=model.fit(train_dataset,epochs=10,validation_data=validation_dataset)
dir_path='' #here we apply another file directory for testing
for i in os.listdir(dir_path):
    img=image.load_img(dir_path+'//'+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val=model.predict(images)
    if val==0:
        print("Covid-19 case")
    else:
        print("Normal case")

