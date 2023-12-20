# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:16:26 2022

@author: WAEL
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
#from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import glob
import pickle
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
X = X / 255.0 
#X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=100, shuffle=True)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.

model.add(Dense(64, input_dim=X.shape[1:]))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
config = model.get_config() # Returns pretty much every information about your model
print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#model.fit(X_train, y_train,
 #         epochs=20,
  #        batch_size=16)
#score = model.evaluate(X_test, y_test, batch_size=16)