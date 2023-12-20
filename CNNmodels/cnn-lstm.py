# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 15:49:11 2021

@author: WAEL
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
import pickle
from keras.preprocessing.image import ImageDataGenerator
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
X = X / 255
classifier = Sequential()

#convolution2D
classifier.add(TimeDistributed(Convolution2D(32,3,3, input_shape=X.shape[1:], activation = 'relu'))) #32 feature detector with 3*3 dimensions, 64*64 is the used format with 3 channel because the image is colored

#adding maxpooling 
classifier.add(TimeDistributed(MaxPooling2D(2, 2)))
#Flattening
classifier.add(TimeDistributed(Flatten()))

classifier.add(TimeDistributed(classifier))
classifier.add(LSTM(units= 20, input_shape = (1,5), return_sequences = True ))
classifier.add(LSTM(units = 20))

#Full Connection
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images

history = classifier.fit_generator(X,
                         steps_per_epoch=2550,
                         epochs=25,
                
                         validation_steps=510)