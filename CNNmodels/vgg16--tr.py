# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 22:05:25 2022

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
train_path = r'C:\Users\WAEL\Desktop\train\databaseOut\train'
valid_path = r'C:\Users\WAEL\Desktop\train\databaseOut\val'
# re-size all the images to this
vggmodel = VGG16(weights='imagenet', include_top=True)
IMAGE_SIZE = [224, 224]
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False
  
folders = glob(r'c:\\train')

x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='sigmoid')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()
model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
r = model.fit_generator(
  train_path,
  validation_data=valid_path,
  epochs=5,
  steps_per_epoch=len(train_path),
  validation_steps=len(valid_path)
)
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
model.save('facefeatures_new_model.h5')

