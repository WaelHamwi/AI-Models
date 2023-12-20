# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 16:46:36 2021

@author: WAEL
"""
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
#mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
#(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
x_train = x_train/255.0
x_test = x_test/255.0
X= X / 255
print(x_train.shape)
print(x_train[0].shape)

model = Sequential()
model.add(LSTM(128, input_shape=X.shape[1:] , activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='softmax'))
config = model.get_config() # Returns pretty much every information about your model
print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

#model.fit(x_train,y_train,epochs=3,validation_data=(x_test, y_test))
model.fit(X,y, epochs=20, validation_split=0.2)